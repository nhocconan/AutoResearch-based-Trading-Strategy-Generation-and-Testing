#!/usr/bin/env python3
"""
Experiment #450: 1d KAMA Trend + Weekly Bias + RSI Pullback + ATR Stop
Hypothesis: Daily timeframe with weekly HTF bias provides cleaner signals than lower TFs.
KAMA (Kaufman Adaptive MA) adapts to volatility - fast in trends, slow in ranges.
This should reduce whipsaw compared to fixed EMAs while maintaining trend capture.
RSI 40-60 pullback zone (not extremes) ensures we enter on dips in trends.
3.5*ATR stoploss appropriate for daily volatility. Multiple entry paths ensure >=10 trades.
Timeframe: 1d (REQUIRED), HTF: 1w via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_weekly_bias_rsi_pullback_atr_v1"
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

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Calculate Kaufman Adaptive Moving Average.
    KAMA adapts to market noise - moves fast in trends, slow in ranges.
    Efficiency Ratio (ER) measures trend direction vs noise.
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Change = absolute price change over period
    change = np.abs(close - np.roll(close, period))
    change[:period] = np.nan
    
    # Sum of absolute differences (volatility/noise)
    volatility = np.zeros(n)
    for i in range(period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-period:i+1])))
    
    # Efficiency Ratio (ER) = trend / noise
    er = np.zeros(n)
    er[:] = np.nan
    mask = volatility > 0
    er[mask] = change[mask] / volatility[mask]
    er = np.clip(er, 0, 1)
    
    # Smoothing constant
    fast_sc = (2 / (fast + 1))
    slow_sc = (2 / (slow + 1))
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama[period] = close[period]
    for i in range(period + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    return kama

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_slope(values, lookback=5):
    """Calculate slope of values over lookback period."""
    n = len(values)
    slope = np.zeros(n)
    slope[:] = np.nan
    for i in range(lookback, n):
        if not np.isnan(values[i]) and not np.isnan(values[i - lookback]):
            slope[i] = (values[i] - values[i - lookback]) / lookback
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    kama_1w = calculate_kama(df_1w['close'].values, 10)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    kama_1d = calculate_kama(close, 10)
    kama_1d_fast = calculate_kama(close, 5)
    rsi = calculate_rsi(close, 14)
    kama_slope = calculate_slope(kama_1d, lookback=5)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_1w_aligned[i]) or np.isnan(kama_1d[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(kama_slope[i]):
            signals[i] = 0.0
            continue
        
        # Weekly trend bias (HTF)
        weekly_bullish = close[i] > kama_1w_aligned[i]
        weekly_bearish = close[i] < kama_1w_aligned[i]
        
        # 1d KAMA trend
        kama_1d_bullish = close[i] > kama_1d[i]
        kama_1d_bearish = close[i] < kama_1d[i]
        kama_rising = kama_slope[i] > 0
        kama_falling = kama_slope[i] < 0
        
        # Fast KAMA crossover
        fast_above_slow = kama_1d_fast[i] > kama_1d[i]
        fast_below_slow = kama_1d_fast[i] < kama_1d[i]
        
        # RSI pullback zones (entry on dips in trend, not extremes)
        rsi_pullback_long = rsi[i] > 35 and rsi[i] < 55
        rsi_pullback_short = rsi[i] > 45 and rsi[i] < 65
        rsi_neutral_long = rsi[i] > 40 and rsi[i] < 60
        rsi_neutral_short = rsi[i] > 40 and rsi[i] < 60
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        # Path 1: Weekly bullish + 1d bullish + RSI pullback + KAMA rising
        if weekly_bullish and kama_1d_bullish and rsi_pullback_long and kama_rising:
            new_signal = SIZE_ENTRY
        # Path 2: Weekly bullish + Fast KAMA above slow + RSI neutral
        elif weekly_bullish and fast_above_slow and rsi_neutral_long:
            new_signal = SIZE_ENTRY
        # Path 3: 1d bullish + KAMA rising + RSI 40-50 (deeper pullback)
        elif kama_1d_bullish and kama_rising and rsi[i] > 40 and rsi[i] < 50:
            new_signal = SIZE_ENTRY
        # Path 4: Weekly bullish + 1d bullish + Fast KAMA crossover up
        elif weekly_bullish and kama_1d_bullish and fast_above_slow and kama_1d_fast[i] > kama_1d_fast[i-1]:
            new_signal = SIZE_ENTRY
        # Path 5: Price above both KAMA + RSI 45-55 (consolidation breakout)
        elif close[i] > kama_1d[i] and close[i] > kama_1w_aligned[i] and rsi[i] > 45 and rsi[i] < 55:
            new_signal = SIZE_ENTRY
        # Path 6: Weekly bullish + KAMA rising (simple trend follow)
        elif weekly_bullish and kama_rising and rsi[i] > 45 and rsi[i] < 65:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        # Path 1: Weekly bearish + 1d bearish + RSI pullback + KAMA falling
        if weekly_bearish and kama_1d_bearish and rsi_pullback_short and kama_falling:
            new_signal = -SIZE_ENTRY
        # Path 2: Weekly bearish + Fast KAMA below slow + RSI neutral
        elif weekly_bearish and fast_below_slow and rsi_neutral_short:
            new_signal = -SIZE_ENTRY
        # Path 3: 1d bearish + KAMA falling + RSI 50-60 (rally short)
        elif kama_1d_bearish and kama_falling and rsi[i] > 50 and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Path 4: Weekly bearish + 1d bearish + Fast KAMA crossover down
        elif weekly_bearish and kama_1d_bearish and fast_below_slow and kama_1d_fast[i] < kama_1d_fast[i-1]:
            new_signal = -SIZE_ENTRY
        # Path 5: Price below both KAMA + RSI 45-55 (consolidation breakdown)
        elif close[i] < kama_1d[i] and close[i] < kama_1w_aligned[i] and rsi[i] > 45 and rsi[i] < 55:
            new_signal = -SIZE_ENTRY
        # Path 6: Weekly bearish + KAMA falling (simple trend follow)
        elif weekly_bearish and kama_falling and rsi[i] > 35 and rsi[i] < 55:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (3.5*ATR for 1d timeframe)
            current_stop = highest_close - 3.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 3.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (3.5*ATR for 1d timeframe)
            current_stop = lowest_close + 3.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 3.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 3.5 * atr[i] if position_side > 0 else close[i] + 3.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 3.5 * atr[i] if position_side > 0 else close[i] + 3.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals