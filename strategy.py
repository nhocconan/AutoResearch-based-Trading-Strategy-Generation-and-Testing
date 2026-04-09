#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and ATR volatility filter
# In bull/bear markets: breakout catches strong moves, volume filter ensures conviction
# ATR filter avoids low-volatility false breakouts. Works in ranging markets by requiring
# both price breakout AND volume expansion. Discrete sizing 0.25 limits trades to ~20-50/year
# to minimize fee drag while maintaining edge in both bull and bear regimes.

name = "4h_1d_donchian_breakout_volume_atr_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ATR(14)
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    
    # Calculate 1d average volume (20-period)
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align 1d indicators to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(avg_volume_1d_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 1d average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume_1d_aligned[i]
        
        # ATR filter: require sufficient volatility (ATR > 0.5% of price)
        atr_filter = atr_1d_aligned[i] > 0.005 * close[i]
        
        if position == 1:  # Long position
            # Exit if price breaks below Donchian low or conditions deteriorate
            if close[i] < lowest_low[i] or not (volume_confirmed and atr_filter):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price breaks above Donchian high or conditions deteriorate
            if close[i] > highest_high[i] or not (volume_confirmed and atr_filter):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long on Donchian high breakout with volume and ATR confirmation
            if close[i] > highest_high[i] and volume_confirmed and atr_filter:
                position = 1
                signals[i] = 0.25
            # Enter short on Donchian low breakout with volume and ATR confirmation
            elif close[i] < lowest_low[i] and volume_confirmed and atr_filter:
                position = -1
                signals[i] = -0.25
    
    return signals