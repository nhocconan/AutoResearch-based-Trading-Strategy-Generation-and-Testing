#!/usr/bin/env python3
"""
Experiment #227: 12h KAMA Adaptive Trend with Daily HMA Filter
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise better than EMA/HMA,
making it ideal for 2025 bear/range markets. KAMA flattens in choppy conditions and trends
sharply in directional moves. Entry on KAMA slope change + price crossover. Daily HMA provides
macro trend bias (only long when price > 1d HMA). Simpler than Donchian/Supertrend, generates
more trades by avoiding over-filtering. ATR stop at 3.0x for wider breathing room.
Position sizing: 0.25 entry, 0.125 half at 2R. Target: Beat Sharpe=0.499 with more consistent trades.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_daily_hma_rsi_atr_v1"
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

def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
    """
    Calculate Kaufman Adaptive Moving Average.
    KAMA adapts to market noise - flattens in chop, trends in direction.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Efficiency Ratio (ER)
    change = np.abs(close - np.roll(close, er_period))
    change[:er_period] = np.abs(close[:er_period] - close[0])
    volatility = np.abs(np.diff(close, prepend=close[0]))
    vol_sum = pd.Series(volatility).rolling(window=er_period, min_periods=1).sum().values
    
    er = np.zeros(n)
    mask = vol_sum > 0
    er[mask] = change[mask] / vol_sum[mask]
    er = np.clip(er, 0, 1)
    
    # Smoothing Constant
    fast_tc = 2 / (fast_sc + 1)
    slow_tc = 2 / (slow_sc + 1)
    sc = (er * (fast_tc - slow_tc) + slow_tc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    kama = calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30)
    
    # KAMA slope (rate of change)
    kama_slope = np.diff(kama, prepend=kama[0])
    
    # Previous values for crossover detection
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    prev_kama = np.roll(kama, 1)
    prev_kama[0] = kama[0]
    prev_kama_slope = np.roll(kama_slope, 1)
    prev_kama_slope[0] = kama_slope[0]
    
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
    
    for i in range(50, n):
        # HTF trend filter (daily HMA)
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        
        # KAMA signals
        kama_bullish = kama_slope[i] > 0
        kama_bearish = kama_slope[i] < 0
        kama_slope_increasing = kama_slope[i] > prev_kama_slope[i]
        kama_slope_decreasing = kama_slope[i] < prev_kama_slope[i]
        
        # Price vs KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # Price crossover KAMA
        cross_above = prev_close[i] <= prev_kama[i] and close[i] > kama[i]
        cross_below = prev_close[i] >= prev_kama[i] and close[i] < kama[i]
        
        # RSI confirmation (not strict filter)
        rsi_bullish = rsi[i] > 40
        rsi_bearish = rsi[i] < 60
        rsi_not_extreme = 25 < rsi[i] < 75
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: Price crosses above KAMA + KAMA slope turning up + daily bullish
        if cross_above and kama_slope_increasing and daily_bullish:
            new_signal = SIZE_ENTRY
        # Secondary: Price above KAMA + KAMA slope positive + RSI confirmation
        elif price_above_kama and kama_bullish and daily_bullish and rsi_bullish:
            # Enter on pullback that holds above KAMA
            if prev_close[i] < kama[i] * 1.01:  # near KAMA
                new_signal = SIZE_ENTRY
        
        # === SHORT ENTRY ===
        # Primary: Price crosses below KAMA + KAMA slope turning down + daily bearish
        if cross_below and kama_slope_decreasing and daily_bearish:
            new_signal = -SIZE_ENTRY
        # Secondary: Price below KAMA + KAMA slope negative + RSI confirmation
        elif price_below_kama and kama_bearish and daily_bearish and rsi_bearish:
            # Enter on pullback that holds below KAMA
            if prev_close[i] > kama[i] * 0.99:  # near KAMA
                new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (3.0*ATR from highest)
            current_stop = highest_close - 3.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 3.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (3.0*ATR from lowest)
            current_stop = lowest_close + 3.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 3.0 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 3.0 * atr[i] if position_side > 0 else close[i] + 3.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 3.0 * atr[i] if position_side > 0 else close[i] + 3.0 * atr[i]
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