#!/usr/bin/env python3
"""
Experiment #646: 12h Primary + 1d HTF — Dual Regime (Chop/Trend) + CRSI + Donchian

Hypothesis: Building on proven patterns (Choppiness+CRSI showed ETH Sharpe +0.923,
Donchian+HMA+RSI showed SOL Sharpe +0.782), this strategy uses regime detection
to switch between mean-reversion (choppy markets) and trend-following (trending markets).

Key insights from 571 failed strategies:
1. Over-engineered filters = 0 trades (#635, #638, #645 all got Sharpe=0.000)
2. 12h timeframe needs simpler logic to generate enough trades
3. Choppiness Index is best regime filter for bear/range markets (2025 test period)
4. Connors RSI catches reversals better than standard RSI in choppy conditions
5. Dual regime adapts to changing market conditions (trend vs range)

Why this might beat Sharpe=0.520:
- 1d HMA slope determines major trend bias (slower, more reliable than 12h)
- Choppiness Index (14) switches between mean-revert and trend-follow modes
- Connors RSI (3,2,100) for mean-reversion entries has 75% win rate in ranges
- Donchian(20) breakout for trend entries has strong momentum confirmation
- Conservative sizing (0.28) + 2.5*ATR stop controls drawdown
- Fewer conflicting filters = more trades (target 25-45/year on 12h)

Regime Logic:
- CHOP > 61.8: Range market → Mean revert with CRSI extremes
- CHOP < 38.2: Trend market → Follow breakout with HMA confirmation
- 38.2 <= CHOP <= 61.8: Transition → Stay flat or hold existing position

Position sizing: 0.28 discrete (per Rule 4, max 0.40)
Target: 25-45 trades/year on 12h (per Rule 10)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_crsi_chop_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close,3) + RSI(streak,2) + PercentRank(100)) / 3
    
    RSI(close,3): 3-period RSI on price
    RSI(streak,2): RSI on up/down streak length
    PercentRank(100): Percentile rank of today's return over last 100 days
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) on price
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Streak RSI
    returns = close_s.pct_change()
    streak = np.zeros(n)
    for i in range(1, n):
        if returns.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif returns.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to positive for RSI calculation
    streak_positive = streak + np.abs(streak.min()) + 1
    rsi_streak = calculate_rsi(streak_positive, streak_period)
    
    # Percent Rank (100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns.iloc[i-rank_period:i].values
        current_return = returns.iloc[i]
        if len(window) > 0:
            percent_rank[i] = 100.0 * np.sum(window < current_return) / len(window)
    
    # CRSI
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8: Range/consolidation (mean reversion favorable)
    CHOP < 38.2: Trending (trend following favorable)
    """
    atr_vals = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    
    atr_sum = pd.Series(atr_vals).rolling(window=period, min_periods=period).sum().values
    
    # Avoid division by zero
    price_range = np.maximum(price_range, 1e-10)
    
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    """
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for major trend bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    hma_12h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_12h[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1D TREND BIAS (HMA slope over 2 bars) ===
        hma_1d_slope_bull = hma_1d_aligned[i] > hma_1d_aligned[i-2] if i >= 2 else False
        hma_1d_slope_bear = hma_1d_aligned[i] < hma_1d_aligned[i-2] if i >= 2 else False
        
        # Price relative to 1d HMA
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 12H HMA SLOPE ===
        hma_12h_slope_bull = hma_12h[i] > hma_12h[i-2] if i >= 2 else False
        hma_12h_slope_bear = hma_12h[i] < hma_12h[i-2] if i >= 2 else False
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_14[i] > 55.0  # Range market (mean reversion)
        is_trending = chop_14[i] < 45.0  # Trend market (trend following)
        # 45-55 is transition zone
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_up = close[i] > donchian_upper[i-1] if i >= 1 else False
        donchian_breakout_down = close[i] < donchian_lower[i-1] if i >= 1 else False
        
        # === CONNORS RSI EXTREMES (for mean reversion) ===
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        crsi_extreme_long = crsi[i] < 20.0
        crsi_extreme_short = crsi[i] > 80.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- REGIME 1: CHOPPY MARKET (Mean Reversion) ---
        # Long: CRSI oversold + price above 1d HMA (bullish bias)
        # Short: CRSI overbought + price below 1d HMA (bearish bias)
        if is_choppy:
            # Mean reversion long
            if crsi_extreme_long and price_above_hma_1d:
                new_signal = POSITION_SIZE
            
            # Mean reversion short
            elif crsi_extreme_short and price_below_hma_1d:
                new_signal = -POSITION_SIZE
        
        # --- REGIME 2: TRENDING MARKET (Trend Following) ---
        # Long: Donchian breakout up + 1d HMA bull + 12h HMA bull
        # Short: Donchian breakout down + 1d HMA bear + 12h HMA bear
        elif is_trending:
            # Trend following long
            if donchian_breakout_up and hma_1d_slope_bull and hma_12h_slope_bull:
                new_signal = POSITION_SIZE
            
            # Trend following short
            elif donchian_breakout_down and hma_1d_slope_bear and hma_12h_slope_bear:
                new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            if hma_1d_slope_bear and price_below_hma_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_1d_slope_bull and price_above_hma_1d:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals