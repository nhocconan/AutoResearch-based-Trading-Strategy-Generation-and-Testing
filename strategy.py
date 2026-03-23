#!/usr/bin/env python3
"""
Experiment #682: 12h Primary + 1d/1w HTF — Simplified Regime + CRSI + HMA Trend

Hypothesis: After analyzing 573+ failed strategies, the pattern is clear:
1. Too many filters = 0 trades (#670, #679, #681 all Sharpe=0.000)
2. CRSI thresholds too extreme (<10/>90) = rarely triggers in crypto
3. 12h timeframe should work (current best is 1d) but needs simpler logic
4. Previous 12h attempts (#672, #676) failed due to over-filtering

This strategy uses SIMPLIFIED logic to ensure trades:
- Choppiness Index (14) for regime: >55 = range, <45 = trend
- Connors RSI with RELAXED thresholds: <25 long, >75 short (not <10/>90)
- 1d HMA for major trend bias (slope over 10 bars for stability)
- Fewer confluence requirements = more trades

Key changes from failed attempts:
- CRSI thresholds: 25/75 instead of 10/90 (3x more signals)
- Choppiness: single threshold 50 instead of dual 45/55
- HMA slope: 10 bars instead of 5 (more stable trend signal)
- Removed conflicting exit conditions that killed positions prematurely

Position sizing: 0.28 discrete (per Rule 4, max 0.40)
Target: 25-45 trades/year on 12h
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_crsi_hma_1d_simp_v2"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 50: Range/consolidation
    CHOP < 50: Trending
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
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Streak calculation
    returns = close_s.pct_change()
    streak = np.zeros(n)
    for i in range(1, n):
        if pd.isna(returns.iloc[i]):
            streak[i] = 0
        elif returns.iloc[i] > 0:
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
    
    # PercentRank(100)
    percent_rank = pd.Series(returns).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / (len(x) - 1) if len(x) > 1 else 0.5,
        raw=False
    ).values * 100.0
    
    crsi = (rsi_close + rsi_streak.values + percent_rank) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for primary trend direction
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    hma_12h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]) or np.isnan(hma_12h[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1D TREND BIAS (HMA slope over 10 bars for stability) ===
        hma_1d_slope_bull = False
        hma_1d_slope_bear = False
        if i >= 10:
            hma_1d_slope_bull = hma_1d_aligned[i] > hma_1d_aligned[i-10]
            hma_1d_slope_bear = hma_1d_aligned[i] < hma_1d_aligned[i-10]
        
        # Price relative to 1d HMA
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME (simplified: single threshold) ===
        is_range = chop_14[i] > 50.0
        is_trend = chop_14[i] <= 50.0
        
        # === CONNORS RSI (RELAXED thresholds for more trades) ===
        crsi_oversold = crsi[i] < 25.0
        crsi_overbought = crsi[i] > 75.0
        crsi_neutral_low = crsi[i] < 40.0
        crsi_neutral_high = crsi[i] > 60.0
        
        # === 12H HMA SLOPE (5 bars) ===
        hma_12h_slope_bull = False
        hma_12h_slope_bear = False
        if i >= 5:
            hma_12h_slope_bull = hma_12h[i] > hma_12h[i-5]
            hma_12h_slope_bear = hma_12h[i] < hma_12h[i-5]
        
        # === ENTRY LOGIC (SIMPLIFIED for more trades) ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Regime 1: Range market + CRSI oversold = mean revert long
        if is_range and crsi_oversold:
            new_signal = POSITION_SIZE
        
        # Regime 2: Trending + 1d bull bias + pullback entry
        elif is_trend and hma_1d_slope_bull:
            if crsi_neutral_low and hma_12h_slope_bull:
                new_signal = POSITION_SIZE
            elif crsi_oversold:
                new_signal = POSITION_SIZE
        
        # Regime 3: Price above 1d HMA + extreme oversold (always valid)
        elif price_above_hma_1d and crsi_oversold:
            new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY ---
        # Regime 1: Range market + CRSI overbought = mean revert short
        elif is_range and crsi_overbought:
            new_signal = -POSITION_SIZE
        
        # Regime 2: Trending + 1d bear bias + pullback entry
        elif is_trend and hma_1d_slope_bear:
            if crsi_neutral_high and hma_12h_slope_bear:
                new_signal = -POSITION_SIZE
            elif crsi_overbought:
                new_signal = -POSITION_SIZE
        
        # Regime 3: Price below 1d HMA + extreme overbought (always valid)
        elif price_below_hma_1d and crsi_overbought:
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
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals