#!/usr/bin/env python3
"""
Experiment #653: 1d Primary + 1w HTF — Simplified Dual Regime (Chop + CRSI + HMA)

Hypothesis: After 574 failed strategies, the clearest pattern is:
1. 1d primary timeframe works best (current best Sharpe=0.520 uses 1d)
2. Too many filters = 0 trades (#645, #648, #650 all Sharpe=0.000)
3. CRSI extremes (<10/>90) have 75% win rate but need regime filter
4. Choppiness Index correctly identifies range vs trend regimes
5. 1w HMA provides major trend bias without over-filtering

This strategy SIMPLIFIES #651 (4h) to 1d primary:
- Fewer regime filters = more trades (critical for passing minimum trade count)
- CRSI thresholds relaxed slightly (<20/>80 instead of <10/>90) to catch more entries
- Choppiness only has 2 regimes (range >55, trend <45) — no ambiguous middle
- 1w HMA slope for major trend direction (slower = fewer whipsaws)
- Asymmetric entries: mean-revert in chop, trend-follow when trending

Why this might beat Sharpe=0.520:
- 1d timeframe = 20-40 trades/year (optimal per Rule 10, less fee drag than 4h)
- Simpler logic = fewer conditions that can all fail simultaneously
- 1w HTF trend filter is slower but more reliable than 1d
- Conservative sizing (0.30) + ATR stop controls drawdown
- Discrete signal levels minimize fee churn

Position sizing: 0.30 discrete (per Rule 4, max 0.40)
Target: 20-40 trades/year on 1d
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_crsi_hma_1w_v2"
timeframe = "1d"
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

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Faster response than EMA with less lag.
    """
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    Interpretation:
    - CHOP > 61.8: Range/consolidation (mean-revert)
    - CHOP < 38.2: Trending (trend-follow)
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0.0, 100.0)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(100)) / 3
    
    Streak RSI: RSI of consecutive up/down days
    PercentRank: Percentile rank of today's return over last 100 days
    
    Entry signals:
    - Long: CRSI < 10 (extreme oversold)
    - Short: CRSI > 90 (extreme overbought)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Streak calculation (consecutive up/down days)
    returns = close_s.pct_change()
    streak = np.zeros(n)
    for i in range(1, n):
        if returns.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif returns.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI(2) on streak
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # PercentRank(100): percentile of today's return over last 100 days
    percent_rank = pd.Series(returns).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / (len(x) - 1) if len(x) > 1 else 0.5,
        raw=False
    ).values * 100.0
    
    # CRSI = average of three components
    crsi = (rsi_close + rsi_streak.values + percent_rank) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for primary trend direction
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    hma_1d = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_1w_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]) or np.isnan(hma_1d[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1W TREND BIAS (HMA slope over 3 bars) ===
        hma_1w_slope_bull = hma_1w_aligned[i] > hma_1w_aligned[i-3] if i >= 3 else False
        hma_1w_slope_bear = hma_1w_aligned[i] < hma_1w_aligned[i-3] if i >= 3 else False
        
        # Price relative to 1w HMA
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_range = chop_14[i] > 55.0  # Range/consolidation
        is_trend = chop_14[i] < 45.0  # Trending
        
        # === CONNORS RSI EXTREMES (relaxed for more trades) ===
        crsi_oversold = crsi[i] < 20.0  # Oversold (relaxed from 10)
        crsi_overbought = crsi[i] > 80.0  # Overbought (relaxed from 90)
        
        # === 1D HMA SLOPE (3 bars) ===
        hma_1d_slope_bull = hma_1d[i] > hma_1d[i-3] if i >= 3 else False
        hma_1d_slope_bear = hma_1d[i] < hma_1d[i-3] if i >= 3 else False
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Regime 1: Range market (CHOP > 55) + CRSI oversold (< 20) = mean revert long
        if is_range and crsi_oversold:
            new_signal = POSITION_SIZE
        
        # Regime 2: Trending market (CHOP < 45) + 1w bull + CRSI pullback
        elif is_trend and hma_1w_slope_bull and price_above_hma_1w:
            if crsi[i] < 45.0:  # Pullback in uptrend
                new_signal = POSITION_SIZE
        
        # Regime 3: 1d HMA bull crossover (trend follow)
        elif hma_1d_slope_bull and crsi[i] < 50.0:
            new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY ---
        # Regime 1: Range market (CHOP > 55) + CRSI overbought (> 80) = mean revert short
        elif is_range and crsi_overbought:
            new_signal = -POSITION_SIZE
        
        # Regime 2: Trending market (CHOP < 45) + 1w bear + CRSI pullback
        elif is_trend and hma_1w_slope_bear and price_below_hma_1w:
            if crsi[i] > 55.0:  # Pullback in downtrend
                new_signal = -POSITION_SIZE
        
        # Regime 3: 1d HMA bear crossover (trend follow)
        elif hma_1d_slope_bear and crsi[i] > 50.0:
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
            if hma_1w_slope_bear and price_below_hma_1w:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_1w_slope_bull and price_above_hma_1w:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
                entry_price = close[i]
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
                entry_price = close[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                entry_price = 0.0
        
        signals[i] = new_signal
    
    return signals