#!/usr/bin/env python3
"""
Experiment #117: 1h KAMA Adaptive Trend + 4h HMA Filter + Volume Confirmation
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise better than EMA.
In ranging markets (2022 crash, 2025 bear), KAMA flattens to avoid whipsaws. In trends,
it accelerates to catch moves. Combine with 4h HMA trend filter (proven in best strategy)
and volume spike confirmation to reduce false breakouts. Use 2.5*ATR stoploss with
1.5R take-profit partial reduction. Position sizing: 0.25 entry, 0.15 after TP.
Timeframe: 1h (mandatory for this experiment) with 4h HTF reference.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_kama_4h_hma_volume_atr_v1"
timeframe = "1h"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = period // 2
    if half < 1:
        half = 1
    sqrt_period = int(np.sqrt(period))
    if sqrt_period < 1:
        sqrt_period = 1
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average.
    KAMA adapts to market noise: flat in ranges, fast in trends.
    ER (Efficiency Ratio) = |Net Change| / Sum of Absolute Changes
    SC (Smoothing Constant) = [ER * (fast_sc - slow_sc) + slow_sc]^2
    """
    close_s = pd.Series(close)
    n = len(close)
    kama = np.zeros(n)
    
    # Efficiency Ratio
    net_change = close_s.diff(er_period).abs()
    sum_changes = close_s.diff().abs().rolling(window=er_period, min_periods=er_period).sum()
    er = net_change / sum_changes.replace(0, np.nan)
    er = er.fillna(0)
    
    # Smoothing Constants
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_volume_spike(volume, period=20):
    """Detect volume spikes (> 2x average volume)."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = vol_s / vol_avg.replace(0, np.nan)
    vol_ratio = vol_ratio.fillna(1)
    return vol_ratio.values

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
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    vol_spike = calculate_volume_spike(volume, 20)
    
    # KAMA adaptive MA
    kama_fast = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_slow = calculate_kama(close, er_period=10, fast_period=5, slow_period=50)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.15
    
    # Track positions for stoploss/takeprofit
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # 4h trend filter (HTF)
        hma_4h_val = hma_4h_aligned[i]
        if np.isnan(hma_4h_val) or hma_4h_val == 0:
            hma_4h_val = close[i]  # fallback if alignment fails
        
        trend_4h_bullish = close[i] > hma_4h_val
        trend_4h_bearish = close[i] < hma_4h_val
        
        # KAMA crossover signals
        kama_cross_long = kama_fast[i] > kama_slow[i] and (i > 0 and kama_fast[i-1] <= kama_slow[i-1])
        kama_cross_short = kama_fast[i] < kama_slow[i] and (i > 0 and kama_fast[i-1] >= kama_slow[i-1])
        
        # KAMA trend state
        kama_trend_long = kama_fast[i] > kama_slow[i]
        kama_trend_short = kama_fast[i] < kama_slow[i]
        
        # KAMA slope confirmation
        kama_slope_long = kama_fast[i] > kama_fast[i-1] if i > 0 else False
        kama_slope_short = kama_fast[i] < kama_fast[i-1] if i > 0 else False
        
        # Volume confirmation (spike > 1.5x average)
        volume_confirmed = vol_spike[i] > 1.5
        
        # RSI filter (avoid extremes)
        rsi_ok_long = rsi[i] < 70
        rsi_ok_short = rsi[i] > 30
        
        # RSI momentum (rising/falling)
        rsi_rising = rsi[i] > rsi[i-1] if i > 0 else False
        rsi_falling = rsi[i] < rsi[i-1] if i > 0 else False
        
        new_signal = 0.0
        
        # LONG ENTRY conditions
        # Condition 1: KAMA cross + 4h bullish + volume spike
        if kama_cross_long and trend_4h_bullish and volume_confirmed:
            new_signal = SIZE_ENTRY
        # Condition 2: KAMA trend + 4h bullish + RSI ok + slope up
        elif kama_trend_long and trend_4h_bullish and rsi_ok_long and kama_slope_long:
            new_signal = SIZE_ENTRY
        # Condition 3: KAMA trend + 4h bullish + RSI rising (momentum)
        elif kama_trend_long and trend_4h_bullish and rsi_rising and rsi_ok_long:
            new_signal = SIZE_ENTRY
        
        # SHORT ENTRY conditions
        # Condition 1: KAMA cross + 4h bearish + volume spike
        if kama_cross_short and trend_4h_bearish and volume_confirmed:
            new_signal = -SIZE_ENTRY
        # Condition 2: KAMA trend + 4h bearish + RSI ok + slope down
        elif kama_trend_short and trend_4h_bearish and rsi_ok_short and kama_slope_short:
            new_signal = -SIZE_ENTRY
        # Condition 3: KAMA trend + 4h bearish + RSI falling (momentum)
        elif kama_trend_short and trend_4h_bearish and rsi_falling and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        
        # Stoploss logic (Rule 6) - check BEFORE updating position
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 1.5R
                initial_risk = 2.5 * atr[i]  # approximate
                profit = close[i] - entry_price
                if profit >= 1.5 * initial_risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 1.5R
                initial_risk = 2.5 * atr[i]  # approximate
                profit = entry_price - close[i]
                if profit >= 1.5 * initial_risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
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