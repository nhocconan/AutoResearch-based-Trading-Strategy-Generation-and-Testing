#!/usr/bin/env python3
"""
Experiment #195: 6h Primary + 12h/1d HTF — Volatility Mean Reversion + Trend Filter

Hypothesis: 6h timeframe captures multi-day swings better than 4h. Previous 6h attempts
failed because they used pure trend-following (doesn't work in 2025 bear market) or
too many conflicting filters (0 trades). This strategy uses:

1. BOLLINGER BAND Z-SCORE: Mean reversion signal when price extends >2.0 std dev
   - Long: z-score < -2.0 (oversold extension)
   - Short: z-score > +2.0 (overbought extension)

2. VOLATILITY SPIKE FILTER: ATR(7)/ATR(30) > 1.5 confirms panic/exhaustion
   - Entry only when vol spike confirms extreme move

3. 1d HMA(50) TREND BIAS: Only take longs when 1d trend neutral/bull
   - Allows counter-trend shorts in bear markets (critical for 2025)

4. 12h CHOPPY INDEX: Avoid entries during transition regimes (CHOP 45-55)

5. CONNORS RSI CONFIRMATION: CRSI < 20 for longs, > 80 for shorts
   - Adds momentum confirmation to mean reversion

Position sizing: 0.25 base, 0.30 for strong confluence
Stoploss: 2.5x ATR trailing stop
Target: 40-60 trades/year, Sharpe > 0.4 (beat current 6h best of 0.399)

Why this differs from failed 6h attempts:
- #187 used simple HMA+RSI pullback (pure trend, failed in bear)
- #191 used ADX+BB+CRSI (too many filters, conflicting signals)
- This uses VOLATILITY + MEAN REVERSION which works in ALL regimes
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_vol_mr_bb_crsi_12h1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands with z-score calculation"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    # Z-score: (price - SMA) / std
    zscore = np.zeros(n)
    zscore[:] = np.nan
    for i in range(period, n):
        if std[i] > 1e-10:
            zscore[i] = (close[i] - sma[i]) / std[i]
    
    return upper, lower, sma, zscore

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = choppy/range bound
    CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    n = len(close)
    if n < pr_period:
        return np.full(n, np.nan)
    
    # RSI(3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_short = np.zeros(n)
    rsi_short[:] = np.nan
    for i in range(rsi_period, n):
        if avg_loss[i] < 1e-10:
            rsi_short[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi_short[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Streak RSI
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(streak_period, n):
        streak_vals = streak[max(0, i-streak_period):i+1]
        up_streak = np.sum(streak_vals > 0)
        down_streak = np.sum(streak_vals < 0)
        total = up_streak + down_streak
        if total > 0:
            streak_rsi[i] = 100.0 * up_streak / total
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank of returns
    returns = np.diff(close) / np.maximum(close[:-1], 1e-10)
    returns = np.concatenate([[0.0], returns])
    
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(pr_period, n):
        window = returns[i-pr_period:i]
        if len(window) > 0:
            count_below = np.sum(window < returns[i])
            percent_rank[i] = 100.0 * count_below / len(window)
    
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 12h CHOP for regime detection
    chop_12h_raw = calculate_choppiness(
        df_12h['high'].values, 
        df_12h['low'].values, 
        df_12h['close'].values, 
        period=14
    )
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h_raw)
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (6h) indicators
    atr = calculate_atr(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    
    bb_upper, bb_lower, bb_sma, bb_zscore = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):  # Start after all indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(bb_zscore[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(chop_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]) or atr_30[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === VOLATILITY SPIKE FILTER ===
        vol_ratio = atr_7[i] / atr_30[i]
        vol_spike = vol_ratio > 1.5  # Volatility elevated
        
        # === BOLLINGER BAND Z-SCORE ===
        zscore = bb_zscore[i]
        bb_extreme_low = zscore < -2.0  # Oversold extension
        bb_extreme_high = zscore > 2.0  # Overbought extension
        
        # === CONNORS RSI CONFIRMATION ===
        crsi_oversold = crsi[i] < 20.0
        crsi_overbought = crsi[i] > 80.0
        
        # === 12h CHOPPY REGIME ===
        chop_12h = chop_12h_aligned[i]
        is_choppy_12h = chop_12h > 50.0  # Prefer mean reversion in choppy 12h
        is_transition_12h = 40.0 <= chop_12h <= 60.0  # Avoid transition
        
        # === 1d HMA TREND BIAS ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRY: BB extreme low + vol spike + CRSI oversold + 1d trend not strongly bear
        if bb_extreme_low and vol_spike and crsi_oversold:
            if htf_1d_bull:
                # Strong long: all conditions + bull trend
                desired_signal = SIZE_STRONG
            elif not htf_1d_bear or is_choppy_12h:
                # Moderate long: neutral/choppy 1d trend allowed in mean reversion
                desired_signal = SIZE_BASE
        
        # SHORT ENTRY: BB extreme high + vol spike + CRSI overbought + 1d trend not strongly bull
        elif bb_extreme_high and vol_spike and crsi_overbought:
            if htf_1d_bear:
                # Strong short: all conditions + bear trend
                desired_signal = -SIZE_STRONG
            elif not htf_1d_bull or is_choppy_12h:
                # Moderate short: neutral/choppy 1d trend allowed
                desired_signal = -SIZE_BASE
        
        # === AVOID TRANSITION REGIME ===
        if is_transition_12h and abs(desired_signal) > 0:
            desired_signal = desired_signal * 0.5  # Reduce size in transition
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif abs(desired_signal) >= SIZE_BASE * 0.4:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals