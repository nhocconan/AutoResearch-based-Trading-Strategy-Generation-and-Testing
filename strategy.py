#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h HMA trend filter + volume spike confirmation
# Donchian breakout captures strong momentum moves in both bull and bear markets
# 12h HMA(21) confirms higher timeframe trend alignment to avoid counter-trend entries
# Volume spike (2x 12h average) confirms breakout authenticity and reduces false signals
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing 0.25
# Works in bull/bear: HMA filter ensures we only take breakouts in direction of 12h trend

name = "4h_12h_donchian_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for HMA and volume
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h HMA(21) - Hull Moving Average
    close_12h = df_12h['close'].values
    def wma(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    def hma(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma_half = wma(values, half_period)
        wma_full = wma(values, period)
        hma_values = 2 * wma_half - wma_full
        # Pad the beginning with NaN
        hma_padded = np.full(len(values), np.nan)
        hma_padded[period-1:] = wma(hma_values, sqrt_period)
        return hma_padded
    
    hma_12h = hma(close_12h, 21)
    
    # Calculate 12h average volume (20-period)
    volume_12h = df_12h['volume'].values
    volume_s_12h = pd.Series(volume_12h)
    avg_volume_12h = volume_s_12h.rolling(window=20, min_periods=20).mean().values
    
    # Align 12h indicators to 4h timeframe (wait for 12h bar close)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    avg_volume_12h_aligned = align_htf_to_ltf(prices, df_12h, avg_volume_12h)
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(hma_12h_aligned[i]) or np.isnan(avg_volume_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 2x 12h average volume
        volume_confirmed = volume[i] > 2.0 * avg_volume_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band
            if close[i] < lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band
            if close[i] > highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: breakout in direction of 12h HMA trend with volume confirmation
            if close[i] > highest_high[i] and hma_12h_aligned[i] > close_12h[-1] if len(close_12h) > 0 else False and volume_confirmed:
                # Only go long if 12h HMA is trending upward (simplified: HMA > previous close)
                # More robust: check if HMA is rising
                if i > 100 and not np.isnan(hma_12h_aligned[i-1]):
                    hma_rising = hma_12h_aligned[i] > hma_12h_aligned[i-1]
                else:
                    hma_rising = True  # Default to true if insufficient data
                if hma_rising:
                    position = 1
                    signals[i] = 0.25
            elif close[i] < lowest_low[i] and hma_12h_aligned[i] < close_12h[-1] if len(close_12h) > 0 else False and volume_confirmed:
                # Only go short if 12h HMA is trending downward
                if i > 100 and not np.isnan(hma_12h_aligned[i-1]):
                    hma_falling = hma_12h_aligned[i] < hma_12h_aligned[i-1]
                else:
                    hma_falling = True  # Default to true if insufficient data
                if hma_falling:
                    position = -1
                    signals[i] = -0.25
    
    return signals