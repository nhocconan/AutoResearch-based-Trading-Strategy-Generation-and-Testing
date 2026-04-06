#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour RSI(14) with 4-hour Supertrend trend filter and volume confirmation.
# Supertrend identifies strong trends (ATR-based), RSI provides mean-reversion entries within trend,
# volume confirms institutional participation. Works in both bull (buy pullbacks in uptrend) and 
# bear (sell bounces in downtrend) markets. Session filter (08-20 UTC) reduces noise.
# Target: 60-150 total trades over 4 years (15-37/year).

name = "exp_13574_1h_rsi14_4h_supertrend_vol_v1"
timeframe = "1h"
leverage = 1.0

# Parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_rsi(close, period):
    """Calculate RSI"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_supertrend(high, low, close, period, multiplier):
    """Calculate Supertrend indicator"""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper = hl2 + (multiplier * atr)
    lower = hl2 - (multiplier * atr)
    
    upper_band = np.zeros_like(close)
    lower_band = np.zeros_like(close)
    upper_band[0] = upper[0]
    lower_band[0] = lower[0]
    
    for i in range(1, len(close)):
        if close[i-1] <= upper_band[i-1]:
            upper_band[i] = min(upper[i], upper_band[i-1])
        else:
            upper_band[i] = upper[i]
            
        if close[i-1] >= lower_band[i-1]:
            lower_band[i] = max(lower[i], lower_band[i-1])
        else:
            lower_band[i] = lower[i]
    
    supertrend = np.zeros_like(close)
    direction = np.ones_like(close, dtype=int)
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(close)):
        if close[i] > upper_band[i-1]:
            direction[i] = 1
        elif close[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    return supertrend, direction

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4-hour data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4-hour Supertrend for trend filter
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    supertrend_4h, trend_direction_4h = calculate_supertrend(high_4h, low_4h, close_4h, SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER)
    supertrend_4h_aligned = align_htf_to_ltf(prices, df_4h, supertrend_4h)
    trend_direction_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_direction_4h)
    
    # Calculate 1-hour indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI
    rsi = calculate_rsi(close, RSI_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for stoploss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Session filter: 08-20 UTC (pre-compute hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(RSI_PERIOD, SUPERTREND_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if Supertrend not available
        if np.isnan(supertrend_4h_aligned[i]) or np.isnan(trend_direction_4h_aligned[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter: 4-hour Supertrend direction
        uptrend = trend_direction_4h_aligned[i] == 1
        downtrend = trend_direction_4h_aligned[i] == -1
        
        # RSI signals: mean reversion within trend
        rsi_oversold = rsi[i] < RSI_OVERSOLD
        rsi_overbought = rsi[i] > RSI_OVERBOUGHT
        
        # Generate signals
        if position == 0:
            if volume_ok and uptrend and rsi_oversold:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif volume_ok and downtrend and rsi_overbought:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals