#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray Index (Bull Power/Bear Power) with EMA trend filter and volume confirmation.
# Bull Power = High - EMA(13), Bear Power = EMA(13) - Low. Measures bull/bear strength relative to trend.
# In trending markets (EMA20 slope), we take trades in direction of trend when power confirms.
# Works in bull markets (buy on bull power) and bear markets (sell on bear power).
# Volume filter ensures institutional participation. Target: 50-150 total trades over 4 years.

name = "elder_ray_ema_volume_6h_v3"
timeframe = "6h"
leverage = 1.0

# Parameters
EMA_SHORT = 13   # For Elder Ray calculation
EMA_TREND = 20   # Trend filter
VOLUME_MA = 20   # Volume confirmation
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def ema_np(array, period):
    """Calculate EMA using numpy for efficiency"""
    return pd.Series(array).ewm(span=period, adjust=False, min_periods=period).mean().values

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
    
    # Load daily data ONCE before loop for Elder Ray and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA for Elder Ray
    close_1d = df_1d['close'].values
    ema_1d = ema_np(close_1d, EMA_SHORT)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate daily EMA for trend filter
    ema_trend = ema_np(close_1d, EMA_TREND)
    ema_trend_aligned = align_htf_to_ltf(prices, df_1d, ema_trend)
    
    # Calculate daily high/low for Elder Ray
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Elder Ray components: Bull Power = High - EMA, Bear Power = EMA - Low
    bull_power = high_1d - ema_1d
    bear_power = ema_1d - low_1d
    
    # Align Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume MA for confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA, min_periods=VOLUME_MA).mean().values
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_SHORT, EMA_TREND, VOLUME_MA, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if np.isnan(ema_trend_aligned[i]) or np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]):
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
        
        # Trend filter: slope of EMA20 (rising/falling)
        if i >= 1:
            ema_now = ema_trend_aligned[i]
            ema_prev = ema_trend_aligned[i-1]
            uptrend = ema_now > ema_prev
            downtrend = ema_now < ema_prev
        else:
            uptrend = False
            downtrend = False
        
        # Elder Ray signals with trend alignment
        long_signal = volume_ok and uptrend and (bull_power_aligned[i] > 0)
        short_signal = volume_ok and downtrend and (bear_power_aligned[i] > 0)
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
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