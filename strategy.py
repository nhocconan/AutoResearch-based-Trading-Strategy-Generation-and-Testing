#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d HMA trend filter and volume confirmation.
# Uses Donchian(20) for clear breakout signals, 1d HMA(21) for smooth trend direction,
# and volume spike (>2x 20-period MA) to confirm institutional participation.
# Discrete sizing 0.25 to manage drawdown. Target: 80-150 total trades over 4 years.
# Works in bull markets via breakouts and in bear markets via short breakdowns with trend filter.

name = "4h_Donchian20_1dHMA21_VolumeSpike_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate 1d HMA(21) trend filter
    def calculate_hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        # WMA function
        def wma(values, window):
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, 'valid') / weights.sum()
        
        wma_half = wma(arr, half_period)
        wma_full = wma(arr, period)
        hma_raw = 2 * wma_half - wma_full
        hma = wma(hma_raw, sqrt_period)
        
        # Pad to original length
        result = np.full_like(arr, np.nan)
        result[period-1:period-1+len(hma)] = hma
        return result
    
    hma_21_1d = calculate_hma(df_1d['close'].values, 21)
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Calculate ATR(14) for 4h data (for stoploss)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0
    lowest_low_since_entry = 0
    
    for i in range(50, n):
        # Skip if HMA is NaN (not enough data yet)
        if np.isnan(hma_21_1d_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian channels for 4h (20-period)
        lookback = 20
        if i < lookback:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        highest_high = np.max(high[i-lookback:i])
        lowest_low = np.min(low[i-lookback:i])
        
        # Volume confirmation: current 4h volume > 2x 20-period MA
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_spike = volume[i] > (2.0 * vol_ma_20)
        else:
            volume_spike = False
        
        # Entry conditions
        # Long: break above Donchian upper with volume spike and above 1d HMA21
        long_entry = (close[i] > highest_high) and volume_spike and (close[i] > hma_21_1d_aligned[i])
        # Short: break below Donchian lower with volume spike and below 1d HMA21
        short_entry = (close[i] < lowest_low) and volume_spike and (close[i] < hma_21_1d_aligned[i])
        
        # Exit conditions (ATR-based trailing stop)
        long_exit = False
        short_exit = False
        
        if position == 1:  # Long position
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            long_exit = close[i] < (highest_high_since_entry - 2.5 * atr[i])
        elif position == -1:  # Short position
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            short_exit = close[i] > (lowest_low_since_entry + 2.5 * atr[i])
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            elif short_entry:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = low[i]
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals