#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian channel breakout with 1-day ATR filter and volume confirmation.
# Uses 1-day Donchian (20) for trend direction and 1-day ATR for volatility filtering.
# Breaks above/below 12-hour Donchian (20) with sufficient volume and ATR > 1-day ATR average
# capture momentum bursts while avoiding chop. Works in bull (breakouts up) and bear (breakdowns down).
# Target: 75-200 total trades over 4 years.

name = "exp_13345_12h_donchian20_1d_atr_vol_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_MA_PERIOD = 10

def calculate_donchian(high, low, period):
    """Calculate Donchian channel upper and lower bands"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Donchian (20) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    upper_1d, lower_1d = calculate_donchian(high_1d, low_1d, DONCHIAN_PERIOD)
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    
    # Calculate daily ATR (14) and its MA for volatility filter
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, ATR_PERIOD)
    atr_ma_1d = pd.Series(atr_1d).rolling(window=ATR_MA_PERIOD, min_periods=ATR_MA_PERIOD).mean().values
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Donchian (20) for entry signals
    upper, lower = calculate_donchian(high, low, DONCHIAN_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # 12h ATR for stoploss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD, ATR_MA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if np.isnan(upper_1d_aligned[i]) or np.isnan(lower_1d_aligned[i]) or np.isnan(atr_ma_1d_aligned[i]):
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
        
        # Trend filter: price above/below daily Donchian midpoint
        mid_1d = (upper_1d_aligned[i] + lower_1d_aligned[i]) / 2
        uptrend = close[i] > mid_1d
        downtrend = close[i] < mid_1d
        
        # Volatility filter: current ATR > daily ATR average
        vol_filter = atr[i] > atr_ma_1d_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Breakout signals using 12h Donchian
        breakout_up = uptrend and vol_filter and volume_ok and (high[i] > upper[i-1])
        breakout_down = downtrend and vol_filter and volume_ok and (low[i] < lower[i-1])
        
        # Generate signals
        if position == 0:
            if breakout_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])  # 2x ATR stop
            elif breakout_down:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])  # 2x ATR stop
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals