"""
SPAN Margin Model — Factor Finance
===================================
Full implementation of NSE SPAN margin calculation.
16 scenario analysis, Black-Scholes option pricing,
portfolio margining, VIX sensitivity.

YouTube : youtube.com/@FactorFinance_2026
GitHub  : https://github.com/FactorFinance/Factor_Finance_Models
===================================
Standard Portfolio Analysis of Risk
Implemented on NSE F&O products

This code walks through SPAN margin calculation
step by step so you understand exactly what
your broker is computing when they show you
a margin requirement.

Run section by section. Read every comment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from dataclasses import dataclass
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# SECTION 1: FOUNDATIONS
# Black-Scholes pricing — the engine under SPAN for options
# ============================================================

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Price a European option using Black-Scholes.
    
    Parameters:
    -----------
    S     : Current underlying price (e.g. Nifty level)
    K     : Strike price
    T     : Time to expiry in years (e.g. 7/365 for weekly)
    r     : Risk-free rate (use RBI repo rate approx)
    sigma : Implied volatility (e.g. 0.15 for 15%)
    
    Returns: Option price in same units as S
    """
    if T <= 0:
        # At expiry — intrinsic value only
        if option_type == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price


def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """Calculate option Greeks — useful context for SPAN."""
    if T <= 0:
        return {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0}
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * norm.pdf(d1) * np.sqrt(T) / 100  # per 1% vol move
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    
    return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta}


# ============================================================
# SECTION 2: THE POSITION CLASS
# Define what you're holding — futures or options
# ============================================================

@dataclass
class Position:
    """
    A single F&O position.
    
    instrument : 'futures' or 'option'
    option_type: 'call' or 'put' (only for options)
    quantity   : Number of lots (negative = short)
    lot_size   : NSE lot size (Nifty = 50)
    strike     : Strike price (options only)
    expiry_days: Days to expiry
    entry_price: Price at which position was entered
    """
    instrument  : str
    quantity    : int          # positive = long, negative = short
    lot_size    : int = 50     # Nifty lot size
    option_type : str = 'call'
    strike      : float = None
    expiry_days : int = 7
    entry_price : float = 0.0


# ============================================================
# SECTION 3: THE HEART OF SPAN
# The 16 scenarios
# ============================================================

def generate_span_scenarios(
    price_scan_range_pct: float,
    vol_scan_range_pct: float
) -> pd.DataFrame:
    """
    Generate the 16 SPAN scenarios.
    
    SPAN defines scenarios as combinations of:
    - Price moves: fractions of the Price Scan Range (PSR)
    - Volatility moves: up or down by the Vol Scan Range (VSR)
    
    The PSR is typically set by the exchange at ~3.5 standard
    deviations of daily returns × sqrt(holding period).
    NSE currently uses approximately 3.5% for Nifty.
    
    Parameters:
    -----------
    price_scan_range_pct : Price Scan Range as % (e.g. 3.5 for 3.5%)
    vol_scan_range_pct   : Volatility Scan Range as % (e.g. 4 for 4%)
    
    Returns: DataFrame with all 16 scenarios
    """
    
    # Price fractions of PSR used in 16 scenarios
    # SPAN uses 0, 1/3, 2/3, and 3/3 (full range)
    # Plus 2× range for extreme scenarios (scenarios 15 & 16)
    price_fractions = [0, 1/3, 2/3, 1, 1, 1, 2/3, 1/3,
                       0, -1/3, -2/3, -1, -1, -1, -2/3, -1/3]
    
    vol_fractions = [+1, -1, +1, -1, +1, -1, +1, -1,
                     +1, -1, +1, -1, +1, -1, +1, -1]
    
    # Scenario 15 and 16 are "extreme" scenarios with 2× price move
    # But only weighted at 35% because they're low probability
    extreme_flags   = [False]*14 + [True, True]
    extreme_weights = [1.0]*14 + [0.35, 0.35]
    
    # The actual extreme scenarios override the above fractions
    price_fractions[14] = +2.0  # Extreme up
    price_fractions[15] = -2.0  # Extreme down
    vol_fractions[14]   = 0     # Vol unchanged for extremes
    vol_fractions[15]   = 0
    
    scenarios = pd.DataFrame({
        'scenario_num'    : range(1, 17),
        'price_fraction'  : price_fractions,
        'vol_fraction'    : vol_fractions,
        'price_move_pct'  : [f * price_scan_range_pct for f in price_fractions],
        'vol_move_pct'    : [f * vol_scan_range_pct for f in vol_fractions],
        'is_extreme'      : extreme_flags,
        'weight'          : extreme_weights
    })
    
    return scenarios


def calculate_position_pnl(
    position: Position,
    current_price: float,
    new_price: float,
    current_vol: float,
    new_vol: float,
    risk_free_rate: float = 0.065
) -> float:
    """
    Calculate P&L for a position under a SPAN scenario.
    
    This is the core calculation SPAN does 16 times per position.
    
    For futures : P&L = (new_price - entry_price) × quantity × lot_size
    For options : P&L = (new_option_price - entry_price) × quantity × lot_size
    
    The option is repriced under the new price AND new volatility
    simultaneously — this is why both price risk and vol risk
    are captured in a single framework.
    """
    T = position.expiry_days / 365
    
    if position.instrument == 'futures':
        # Futures P&L is purely linear in price
        pnl_per_unit = new_price - position.entry_price
        
    elif position.instrument == 'option':
        # Reprice the option under new market conditions
        new_option_price = black_scholes(
            S           = new_price,
            K           = position.strike,
            T           = T,
            r           = risk_free_rate,
            sigma       = new_vol / 100,
            option_type = position.option_type
        )
        pnl_per_unit = new_option_price - position.entry_price
    
    # Total P&L = per unit × lots × lot size
    total_pnl = pnl_per_unit * position.quantity * position.lot_size
    return total_pnl


def run_span_scenarios(
    positions       : List[Position],
    current_price   : float,
    current_vol_pct : float,
    price_scan_range_pct : float = 3.5,
    vol_scan_range_pct   : float = 4.0,
    risk_free_rate  : float = 0.065
) -> Tuple[pd.DataFrame, float]:
    """
    Run all 16 SPAN scenarios on a portfolio of positions.
    
    This is the complete SPAN calculation:
    1. Generate 16 scenarios
    2. For each scenario, calculate portfolio P&L
    3. Apply extreme scenario weights
    4. SPAN margin = worst case loss (most negative P&L)
    
    Returns:
    --------
    scenario_results : DataFrame with P&L for each scenario
    span_margin      : The margin requirement (positive number = ₹)
    """
    scenarios = generate_span_scenarios(
        price_scan_range_pct, vol_scan_range_pct
    )
    
    portfolio_pnls = []
    
    for _, scenario in scenarios.iterrows():
        
        # New price and volatility under this scenario
        new_price = current_price * (1 + scenario['price_move_pct'] / 100)
        new_vol   = current_vol_pct + scenario['vol_move_pct']
        new_vol   = max(new_vol, 1.0)  # Vol can't go negative
        
        # Sum P&L across all positions in portfolio
        portfolio_pnl = sum(
            calculate_position_pnl(
                position        = pos,
                current_price   = current_price,
                new_price       = new_price,
                current_vol     = current_vol_pct,
                new_vol         = new_vol,
                risk_free_rate  = risk_free_rate
            )
            for pos in positions
        )
        
        # Apply extreme scenario weight (scenarios 15 & 16 get 35% weight)
        weighted_pnl = portfolio_pnl * scenario['weight']
        portfolio_pnls.append(weighted_pnl)
    
    scenarios['portfolio_pnl'] = portfolio_pnls
    
    # SPAN margin = worst case loss = most negative scenario
    # If all scenarios are profitable, SPAN margin = 0
    worst_case_pnl = min(portfolio_pnls)
    span_margin = max(-worst_case_pnl, 0)
    
    return scenarios, span_margin


# ============================================================
# SECTION 4: EXPOSURE MARGIN
# NSE mandates an additional buffer on top of SPAN
# ============================================================

def calculate_exposure_margin(
    positions       : List[Position],
    current_price   : float,
    index_futures_pct : float = 3.0,
    stock_futures_pct : float = 5.0
) -> float:
    """
    Exposure margin is NSE's second line of defense.
    
    It's a simpler calculation — a fixed percentage of the
    notional value of the position, added on top of SPAN.
    
    For index futures  : 3% of notional
    For stock futures  : 5% of notional (higher due to single stock risk)
    For options bought : No exposure margin (max loss is premium paid)
    For options sold   : 3-5% of notional
    """
    total_exposure_margin = 0
    
    for pos in positions:
        notional = abs(pos.quantity) * pos.lot_size * current_price
        
        if pos.instrument == 'futures':
            exposure_margin = notional * index_futures_pct / 100
        
        elif pos.instrument == 'option':
            if pos.quantity > 0:
                # Long option — max loss is premium, no exposure margin
                exposure_margin = 0
            else:
                # Short option — exposure margin applies
                exposure_margin = notional * index_futures_pct / 100
        
        total_exposure_margin += exposure_margin
    
    return total_exposure_margin


# ============================================================
# SECTION 5: FULL MARGIN CALCULATION
# Putting it all together
# ============================================================

def calculate_total_margin(
    positions            : List[Position],
    current_price        : float,
    current_vol_pct      : float,
    price_scan_range_pct : float = 3.5,
    vol_scan_range_pct   : float = 4.0,
    risk_free_rate       : float = 0.065,
    verbose              : bool  = True
) -> dict:
    """
    Calculate complete NSE margin requirement for a portfolio.
    
    Total Margin = SPAN Margin + Exposure Margin
    
    Returns dictionary with full breakdown.
    """
    
    # Step 1: Run SPAN scenarios
    scenario_results, span_margin = run_span_scenarios(
        positions            = positions,
        current_price        = current_price,
        current_vol_pct      = current_vol_pct,
        price_scan_range_pct = price_scan_range_pct,
        vol_scan_range_pct   = vol_scan_range_pct,
        risk_free_rate       = risk_free_rate
    )
    
    # Step 2: Calculate exposure margin
    exposure_margin = calculate_exposure_margin(
        positions     = positions,
        current_price = current_price
    )
    
    # Step 3: Total margin
    total_margin = span_margin + exposure_margin
    
    if verbose:
        print("\n" + "="*60)
        print("SPAN MARGIN CALCULATION — Factor Finance")
        print("="*60)
        print(f"\nUnderlying Price  : ₹{current_price:,.0f}")
        print(f"Current Vol (VIX) : {current_vol_pct:.1f}%")
        print(f"Price Scan Range  : ±{price_scan_range_pct:.1f}%")
        print(f"Vol Scan Range    : ±{vol_scan_range_pct:.1f}%")
        
        print(f"\n{'POSITIONS':}")
        print("-"*60)
        for i, pos in enumerate(positions):
            direction = "Long" if pos.quantity > 0 else "Short"
            lots = abs(pos.quantity)
            if pos.instrument == 'futures':
                print(f"  {i+1}. {direction} {lots} lot(s) Nifty Futures "
                      f"@ ₹{pos.entry_price:,.0f}")
            else:
                print(f"  {i+1}. {direction} {lots} lot(s) Nifty "
                      f"{pos.strike:.0f} {pos.option_type.upper()} "
                      f"@ ₹{pos.entry_price:.2f}")
        
        print(f"\n{'16 SCENARIO RESULTS':}")
        print("-"*60)
        print(f"{'Scenario':<10} {'Price Move':>12} {'Vol Move':>10} "
              f"{'P&L':>14} {'Weighted P&L':>14}")
        print("-"*60)
        
        for _, row in scenario_results.iterrows():
            marker = " ← WORST" if row['portfolio_pnl'] == scenario_results['portfolio_pnl'].min() else ""
            extreme = " [35%]" if row['is_extreme'] else ""
            print(f"  {int(row['scenario_num']):<8} "
                  f"{row['price_move_pct']:>+10.2f}%  "
                  f"{row['vol_move_pct']:>+8.1f}%  "
                  f"₹{row['portfolio_pnl']:>12,.0f}"
                  f"{extreme}{marker}")
        
        print("-"*60)
        print(f"\n{'MARGIN SUMMARY':}")
        print("-"*60)
        print(f"  SPAN Margin       : ₹{span_margin:>10,.0f}")
        print(f"  Exposure Margin   : ₹{exposure_margin:>10,.0f}")
        print(f"  {'─'*30}")
        print(f"  TOTAL MARGIN      : ₹{total_margin:>10,.0f}")
        print("="*60)
    
    return {
        'span_margin'      : span_margin,
        'exposure_margin'  : exposure_margin,
        'total_margin'     : total_margin,
        'scenario_results' : scenario_results,
        'worst_scenario'   : scenario_results.loc[
            scenario_results['portfolio_pnl'].idxmin()
        ]
    }


# ============================================================
# SECTION 6: VISUALISATION
# Make the model visual — this becomes your video content
# ============================================================

def plot_span_analysis(result: dict, title: str = ""):
    """
    Visualise the complete SPAN analysis.
    Three charts:
    1. P&L across all 16 scenarios
    2. Margin component breakdown
    3. Scenario heatmap
    """
    
    scenarios = result['scenario_results']
    
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#0D0D0D')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
    
    # ── Chart 1: P&L Bar Chart Across 16 Scenarios ──
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor('#0D0D0D')
    
    colors = ['#FF4444' if p < 0 else '#44FF88' 
              for p in scenarios['portfolio_pnl']]
    bars = ax1.bar(scenarios['scenario_num'],
                   scenarios['portfolio_pnl'],
                   color=colors, alpha=0.85, edgecolor='none')
    
    # Highlight worst case
    worst_idx = scenarios['portfolio_pnl'].idxmin()
    bars[worst_idx].set_color('#FF0000')
    bars[worst_idx].set_alpha(1.0)
    bars[worst_idx].set_edgecolor('#FFaa00')
    bars[worst_idx].set_linewidth(2)
    
    worst_pnl = scenarios['portfolio_pnl'].min()
    ax1.annotate(f'WORST CASE\n₹{worst_pnl:,.0f}\n= SPAN Margin',
                xy=(worst_idx + 1, worst_pnl),
                xytext=(worst_idx + 2.5, worst_pnl * 0.7),
                color='#FFaa00', fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#FFaa00', lw=1.5))
    
    ax1.axhline(y=0, color='#555555', linewidth=1)
    ax1.set_xlabel('SPAN Scenario Number', color='#888888', fontsize=10)
    ax1.set_ylabel('Portfolio P&L (₹)', color='#888888', fontsize=10)
    ax1.set_title(f'P&L Across All 16 SPAN Scenarios — {title}',
                  color='white', fontsize=13, pad=12)
    ax1.tick_params(colors='#888888')
    ax1.spines['bottom'].set_color('#333333')
    ax1.spines['left'].set_color('#333333')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_xticks(range(1, 17))
    
    # Add scenario labels
    for i, (_, row) in enumerate(scenarios.iterrows()):
        if row['is_extreme']:
            ax1.text(row['scenario_num'], 
                    max(scenarios['portfolio_pnl']) * 0.05,
                    'EXT', ha='center', color='#FFaa00', fontsize=7)
    
    # ── Chart 2: Margin Component Breakdown ──
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor('#0D0D0D')
    
    components = ['SPAN\nMargin', 'Exposure\nMargin', 'Total\nMargin']
    values = [result['span_margin'], result['exposure_margin'], 
              result['total_margin']]
    bar_colors = ['#4488FF', '#FF8844', '#FFFFFF']
    
    bars2 = ax2.bar(components, values, color=bar_colors,
                    width=0.5, alpha=0.85, edgecolor='none')
    
    for bar, val in zip(bars2, values):
        ax2.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + result['total_margin'] * 0.02,
                f'₹{val:,.0f}',
                ha='center', va='bottom',
                color='white', fontsize=11, fontweight='bold')
    
    ax2.set_title('Margin Component Breakdown',
                  color='white', fontsize=11, pad=10)
    ax2.set_ylabel('₹', color='#888888')
    ax2.tick_params(colors='#888888')
    ax2.spines['bottom'].set_color('#333333')
    ax2.spines['left'].set_color('#333333')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_ylim(0, max(values) * 1.25)
    
    # ── Chart 3: Scenario Heatmap ──
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor('#0D0D0D')
    
    # Reshape into a grid for the heatmap
    # 4 price levels × 2 vol states × 2 directions ≈ interpretable grid
    pnl_matrix = np.zeros((4, 4))
    
    # Map scenarios to approximate grid positions
    price_steps = [-1, -2/3, -1/3, 0, 1/3, 2/3, 1]
    
    # Use first 14 scenarios for the heatmap (exclude extremes)
    heat_data = scenarios[~scenarios['is_extreme']].copy()
    
    # Create a readable price vs vol grid
    price_bins = pd.cut(heat_data['price_move_pct'], 
                        bins=4, labels=['-Full', '-Half', '+Half', '+Full'])
    vol_bins   = heat_data['vol_fraction'].map({1.0: 'Vol Up', -1.0: 'Vol Down'})
    
    try:
        pivot = heat_data.pivot_table(
            values='portfolio_pnl',
            index='vol_fraction',
            columns=pd.cut(heat_data['price_move_pct'], bins=[-100,-2,-0.1,0.1,2,100],
                          labels=['Down A Lot','Down','Flat','Up','Up A Lot']),
            aggfunc='mean'
        )
        
        im = ax3.imshow(pivot.values, cmap='RdYlGn', aspect='auto')
        
        ax3.set_xticks(range(len(pivot.columns)))
        ax3.set_xticklabels(pivot.columns, color='#888888', fontsize=8)
        ax3.set_yticks(range(len(pivot.index)))
        ax3.set_yticklabels(['Vol Up' if v == 1 else 'Vol Down' 
                            for v in pivot.index], color='#888888', fontsize=9)
        
        for i in range(pivot.values.shape[0]):
            for j in range(pivot.values.shape[1]):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax3.text(j, i, f'₹{val/1000:.0f}K',
                            ha='center', va='center',
                            color='white' if abs(val) > pivot.values.max()*0.5 else 'black',
                            fontsize=8, fontweight='bold')
        
    except Exception:
        ax3.text(0.5, 0.5, 'Heatmap requires\nmore scenario variety',
                ha='center', va='center', color='white', transform=ax3.transAxes)
    
    ax3.set_title('P&L Heatmap\nPrice Move vs Volatility Change',
                  color='white', fontsize=11, pad=10)
    ax3.tick_params(colors='#888888')
    for spine in ax3.spines.values():
        spine.set_color('#333333')
    
    plt.suptitle('SPAN Margin Analysis — Factor Finance',
                 color='white', fontsize=15, y=1.01)
    
    plt.savefig('span_full_analysis.png', dpi=150,
                facecolor='#0D0D0D', bbox_inches='tight')
    plt.show()
    print("Chart saved: span_full_analysis.png")


# ============================================================
# SECTION 7: RUN THE EXAMPLES
# Three examples — learn by comparing them
# ============================================================

def main():
    
    # ── Market Parameters ──
    NIFTY_PRICE     = 22000
    INDIA_VIX       = 14.5   # Approximate current VIX
    RISK_FREE_RATE  = 0.065  # RBI repo rate approximate
    PSR             = 3.5    # Price Scan Range %
    VSR             = 4.0    # Vol Scan Range %
    
    
    print("\n" + "█"*60)
    print("  EXAMPLE 1: NAKED SHORT FUTURES")
    print("  Selling 1 lot Nifty Futures")
    print("  This is your baseline — maximum margin case")
    print("█"*60)
    
    naked_futures = [
        Position(
            instrument  = 'futures',
            quantity    = -1,           # Short 1 lot
            lot_size    = 50,
            entry_price = NIFTY_PRICE
        )
    ]
    
    result_naked = calculate_total_margin(
        positions            = naked_futures,
        current_price        = NIFTY_PRICE,
        current_vol_pct      = INDIA_VIX,
        price_scan_range_pct = PSR,
        vol_scan_range_pct   = VSR,
        verbose              = True
    )
    
    
    print("\n" + "█"*60)
    print("  EXAMPLE 2: HEDGED FUTURES")
    print("  Short Futures + Long Put (same expiry)")
    print("  This is the KEY example — watch the margin DROP")
    print("█"*60)
    
    # Price a 22000 Put to set entry price
    put_price = black_scholes(
        S=NIFTY_PRICE, K=NIFTY_PRICE, T=7/365,
        r=RISK_FREE_RATE, sigma=INDIA_VIX/100,
        option_type='put'
    )
    
    hedged_position = [
        Position(
            instrument  = 'futures',
            quantity    = -1,
            lot_size    = 50,
            entry_price = NIFTY_PRICE
        ),
        Position(
            instrument  = 'option',
            option_type = 'put',
            quantity    = +1,           # Long put as hedge
            lot_size    = 50,
            strike      = NIFTY_PRICE,
            expiry_days = 7,
            entry_price = put_price
        )
    ]
    
    result_hedged = calculate_total_margin(
        positions            = hedged_position,
        current_price        = NIFTY_PRICE,
        current_vol_pct      = INDIA_VIX,
        price_scan_range_pct = PSR,
        vol_scan_range_pct   = VSR,
        verbose              = True
    )
    
    
    print("\n" + "█"*60)
    print("  EXAMPLE 3: HIGH VOLATILITY EVENT")
    print("  Same naked futures position")
    print("  But VIX is now 22 (pre-budget/RBI event)")
    print("  Watch the margin INCREASE without touching position")
    print("█"*60)
    
    result_high_vol = calculate_total_margin(
        positions            = naked_futures,
        current_price        = NIFTY_PRICE,
        current_vol_pct      = 22.0,    # VIX spiked to 22
        price_scan_range_pct = PSR + 1, # Exchange widens PSR too
        vol_scan_range_pct   = VSR + 1,
        verbose              = True
    )
    
    
    # ── Summary Comparison ──
    print("\n" + "="*60)
    print("  THE KEY INSIGHT — MARGIN COMPARISON")
    print("="*60)
    print(f"\n  {'Scenario':<35} {'Total Margin':>15}")
    print(f"  {'─'*50}")
    print(f"  {'Naked Short Futures (Normal Vol)':<35} "
          f"₹{result_naked['total_margin']:>12,.0f}")
    print(f"  {'Hedged (Futures + Long Put)':<35} "
          f"₹{result_hedged['total_margin']:>12,.0f}")
    print(f"  {'Naked Short Futures (High Vol)':<35} "
          f"₹{result_high_vol['total_margin']:>12,.0f}")
    
    saving = result_naked['total_margin'] - result_hedged['total_margin']
    saving_pct = saving / result_naked['total_margin'] * 100
    increase = result_high_vol['total_margin'] - result_naked['total_margin']
    increase_pct = increase / result_naked['total_margin'] * 100
    
    print(f"\n  {'─'*50}")
    print(f"  Hedging saves   : ₹{saving:,.0f} ({saving_pct:.0f}% reduction)")
    print(f"  High vol costs  : ₹{increase:,.0f} more ({increase_pct:.0f}% increase)")
    print(f"\n  This is why your margin changes overnight.")
    print(f"  This is why hedging reduces your requirement.")
    print(f"  SPAN is not arbitrary. It is mathematical.")
    print("="*60)
    
    # Plot the naked futures analysis
    plot_span_analysis(result_naked, title="Naked Short Futures")
    
    
    # ── Bonus: How margin changes as VIX moves ──
    print("\n" + "█"*60)
    print("  BONUS: MARGIN vs VIX LEVEL")
    print("  This shows WHY margin spikes before events")
    print("█"*60)
    
    vix_levels = range(10, 36, 2)
    margins = []
    
    for vix in vix_levels:
        psr = max(3.5, vix / 4)   # PSR widens as VIX rises
        r = calculate_total_margin(
            positions            = naked_futures,
            current_price        = NIFTY_PRICE,
            current_vol_pct      = float(vix),
            price_scan_range_pct = psr,
            vol_scan_range_pct   = VSR,
            verbose              = False
        )
        margins.append(r['total_margin'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0D0D0D')
    ax.set_facecolor('#0D0D0D')
    
    ax.plot(list(vix_levels), margins, color='#FF4444',
            linewidth=2.5, marker='o', markersize=6)
    ax.fill_between(list(vix_levels), margins,
                    alpha=0.15, color='#FF4444')
    
    # Mark key VIX zones
    ax.axvspan(10, 15, alpha=0.05, color='#44FF88')
    ax.axvspan(15, 22, alpha=0.05, color='#FFaa00')
    ax.axvspan(22, 35, alpha=0.05, color='#FF4444')
    
    ax.text(12,   min(margins)*1.01, 'Calm',     color='#44FF88', fontsize=9)
    ax.text(17.5, min(margins)*1.01, 'Elevated', color='#FFaa00', fontsize=9)
    ax.text(26,   min(margins)*1.01, 'Stressed', color='#FF4444', fontsize=9)
    
    ax.set_title(
        'SPAN Margin vs India VIX Level\n'
        'Same Position — Margin Changes as VIX Changes',
        color='white', fontsize=13, pad=12
    )
    ax.set_xlabel('India VIX Level', color='#888888', fontsize=11)
    ax.set_ylabel('Total Margin Required (₹)', color='#888888', fontsize=11)
    ax.tick_params(colors='#888888')
    ax.spines['bottom'].set_color('#333333')
    ax.spines['left'].set_color('#333333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Annotate key insight
    ax.text(0.05, 0.92,
            'You changed nothing.\nThe market changed.\nYour margin changed.',
            transform=ax.transAxes, color='#888888', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='#1A1A1A', alpha=0.8),
            verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('span_margin_vs_vix.png', dpi=150,
                facecolor='#0D0D0D', bbox_inches='tight')
    plt.show()
    print("\nAll charts saved. Upload to GitHub finance-models repository.")


if __name__ == "__main__":
    main()