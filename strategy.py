#!/usr/bin/env python3
"""
Experiment #9563: 4h Donchian Breakout + HMA Trend + Volume Confirmation + ATR Stoploss.
Hypothesis: Donchian channel breakouts provide directional bias, confirmed by HMA trend and volume spikes.
Works in bull markets (breakouts above upper band) and bear markets (breakdowns below lower band).
Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9563_4h_donchian20_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
HMA_FAST = 9
HMA_SLOW = 21
VOLUME_SPIKE_MULTIPLIER = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def hull_moving_average(arr, period):
    """Calculate Hull Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    half = int(period / 2)
    sqrt = int(np.sqrt(period))
    wma1 = pd.Series(arr).rolling(window=half, min_periods=half).mean()
    wma2 = pd.Series(arr).rolling(window=period, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = pd.Series(raw_hma).rolling(window=sqrt, min_periods=sqrt).mean()
    return hma.values

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
    
    # Load HTF data ONCE before loop (1d for trend filter)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for trend filter
    hma_1d = hull_moving_average(df_1d['close'].values, HMA_SLOW)
    hma_1d_prev = np.roll(hma_1d, 1)
    hma_1d_prev[0] = np.nan
    hma_1d_trend = hma_1d > hma_1d_prev  # True = uptrend
    
    # Align 1d HMA trend to 4h timeframe
    hma_1d_trend_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_trend.astype(float))
    
    # Calculate LTF indicators (4h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # HMA for trend confirmation
    hma_fast = hull_moving_average(close, HMA_FAST)
    hma_slow = hull_moving_average(close, HMA_SLOW)
    hma_trend = hma_fast > hma_slow  # True = uptrend
    
    # Volume moving average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, 20) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(hma_1d_trend_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Breakout conditions
        breakout_long = (high[i] >= highest_high[i-1]) and volume_spike  # Break above upper band
        breakout_short = (low[i] <= lowest_low[i-1]) and volume_spike   # Break below lower band
        
        # Trend filters: 1d HMA trend + 4h HMA alignment
        long_trend = hma_1d_trend_aligned[i] and hma_trend[i]
        short_trend = (not hma_1d_trend_aligned[i]) and (not hma_trend[i])
        
        # Entry conditions
        long_entry = breakout_long and long_trend
        short_entry = breakout_short and short_trend
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
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