#!/usr/bin/env python3
"""
Experiment #9718: 1d Donchian Breakout + Volume Spike + Weekly Trend Filter
Hypothesis: Daily Donchian(20) breakouts with volume confirmation and weekly trend filter
(HMA21 > HMA50 for uptrend, HMA21 < HMA50 for downtrend) provide high-probability
trend-following signals. Targets 30-100 total trades over 4 years (7-25/year) to
minimize fee drag while capturing major moves. Works in bull (breakouts above upper band)
and bear (breakdowns below lower band) with trend filter preventing counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9718_1d_donchian_breakout_volume_weekly_trend_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SPIKE_MULTIPLIER = 2.0
HMA_FAST = 21
HMA_SLOW = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_hma(arr, period):
    """Hull Moving Average"""
    n = int(period)
    if n < 1:
        return arr
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    # WMA of half period
    weights_half = np.arange(1, half + 1)
    wma_half = np.convolve(arr, weights_half, 'valid') / weights_half.sum()
    
    # WMA of full period
    weights_full = np.arange(1, n + 1)
    wma_full = np.convolve(arr, weights_full, 'valid') / weights_full.sum()
    
    # Raw HMA
    hma_raw = 2 * wma_half - wma_full
    
    # Final WMA of sqrt period
    weights_sqrt = np.arange(1, sqrt_n + 1)
    hma = np.convolve(hma_raw, weights_sqrt, 'valid') / weights_sqrt.sum()
    
    # Pad to original length
    hma_padded = np.full_like(arr, np.nan)
    hma_padded[n-1:] = hma
    return hma_padded

def calculate_atr(high, low, close, period):
    """ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (weekly for trend filter)
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly HMA for trend filter
    weekly_close = df_weekly['close'].values
    hma_fast = calculate_hma(weekly_close, HMA_FAST)
    hma_slow = calculate_hma(weekly_close, HMA_SLOW)
    
    # Align weekly HMA to daily timeframe
    hma_fast_aligned = align_htf_to_ltf(prices, df_weekly, hma_fast)
    hma_slow_aligned = align_htf_to_ltf(prices, df_weekly, hma_slow)
    
    # Calculate daily indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
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
        # Skip if weekly trend data not available
        if np.isnan(hma_fast_aligned[i]) or np.isnan(hma_slow_aligned[i]):
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
        
        # Weekly trend filter: HMA21 > HMA50 = uptrend, HMA21 < HMA50 = downtrend
        uptrend = hma_fast_aligned[i] > hma_slow_aligned[i]
        downtrend = hma_fast_aligned[i] < hma_slow_aligned[i]
        
        # Volume spike confirmation
        volume_spike = volume[i] > (volume_ma[i] * VOLUME_SPIKE_MULTIPLIER) if not np.isnan(volume_ma[i]) else False
        
        # Breakout conditions
        breakout_up = high[i] >= highest_high[i]  # Price breaks above Donchian upper band
        breakout_down = low[i] <= lowest_low[i]   # Price breaks below Donchian lower band
        
        # Entry conditions
        long_entry = uptrend and volume_spike and breakout_up
        short_entry = downtrend and volume_spike and breakout_down
        
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