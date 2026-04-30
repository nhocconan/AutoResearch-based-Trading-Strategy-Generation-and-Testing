#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h HMA(21) trend filter and volume confirmation
# Targets 75-200 total trades over 4 years (19-50/year) on 4h timeframe.
# Donchian breakouts capture momentum moves; 12h HMA filters for higher-timeframe trend alignment;
# volume spike (2.0x 20-period average) confirms breakout validity.
# Works in bull markets via breakout longs with uptrend filter, in bear markets via breakout shorts with downtrend filter.
# Discrete sizing 0.25 minimizes fee churn while maintaining adequate position size.

name = "4h_Donchian20_Breakout_12hHMA21_VolumeSpike_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Load 12h data ONCE before loop for HMA calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate 12h HMA(21) - Hull Moving Average
    def calculate_hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        # WMA calculation
        def wma(values, window):
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, 'valid') / weights.sum()
        
        wma_half = np.array([np.nan] * (len(arr) - half_period + 1))
        wma_full = np.array([np.nan] * (len(arr) - period + 1))
        
        for i in range(len(arr) - half_period + 1):
            wma_half[i] = np.dot(arr[i:i+half_period], np.arange(1, half_period+1)) / (half_period * (half_period + 1) / 2)
        for i in range(len(arr) - period + 1):
            wma_full[i] = np.dot(arr[i:i+period], np.arange(1, period+1)) / (period * (period + 1) / 2)
        
        raw_hma = 2 * wma_half - wma_full
        hma = np.array([np.nan] * (len(arr) - sqrt_period + 1))
        for i in range(len(raw_hma)):
            hma[i + sqrt_period - 1] = np.dot(raw_hma[i:i+sqrt_period], np.arange(1, sqrt_period+1)) / (sqrt_period * (sqrt_period + 1) / 2)
        
        # Pad with NaN to match original length
        result = np.full_like(arr, np.nan)
        result[period-1:] = hma
        return result
    
    hma_12h = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 20  # warmup for Donchian channels
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(hma_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_highest = highest_20[i]
        curr_lowest = lowest_20[i]
        curr_hma = hma_12h_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish breakout: price breaks above highest_20 AND 12h HMA is rising (uptrend)
                if curr_close > curr_highest and curr_hma > hma_12h_aligned[i-1]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish breakout: price breaks below lowest_20 AND 12h HMA is falling (downtrend)
                elif curr_close < curr_lowest and curr_hma < hma_12h_aligned[i-1]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below lowest_20 (breakdown of support)
            if curr_close < curr_lowest:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above highest_20 (breakout of resistance)
            if curr_close > curr_highest:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals