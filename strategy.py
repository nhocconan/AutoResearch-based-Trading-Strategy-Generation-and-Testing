#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h ATR filter and volume confirmation
# In trending markets: breakout above/below Donchian channels with volume > 1.5x average
# Uses ATR to filter low-volatility choppy periods where breakouts fail
# Discrete position sizing 0.25 to limit trades to ~12-37/year and reduce fee drag
# Works in bull/bear markets: breakout captures trends, ATR filter avoids false breakouts in ranging markets

name = "6h_12h_donchian_breakout_atr_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h ATR(14) for volatility filter
    tr1 = np.abs(high_12h[1:] - low_12h[:-1])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
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
    
    atr_12h = wilders_smoothing(tr, 14)
    
    # Calculate 12h average ATR (20-period) for normalization
    atr_s_12h = pd.Series(atr_12h)
    avg_atr_12h = atr_s_12h.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h average volume (20-period)
    volume_s_12h = pd.Series(volume_12h)
    avg_volume_12h = volume_s_12h.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 6h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align 12h indicators to 6h timeframe
    avg_atr_12h_aligned = align_htf_to_ltf(prices, df_12h, avg_atr_12h)
    avg_volume_12h_aligned = align_htf_to_ltf(prices, df_12h, avg_volume_12h)
    
    # Pre-compute confirmation arrays
    volume_confirmed = volume > 1.5 * avg_volume_12h_aligned
    atr_filter = avg_atr_12h_aligned > 0  # Only trade when ATR is valid (volatility present)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(avg_atr_12h_aligned[i]) or np.isnan(avg_volume_12h_aligned[i]) or
            np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long if price breaks below Donchian low or volatility drops
            if close[i] < lowest_low[i] or not atr_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price breaks above Donchian high or volatility drops
            if close[i] > highest_high[i] or not atr_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long on breakout above Donchian high with volume confirmation and sufficient volatility
            if close[i] > highest_high[i] and volume_confirmed[i] and atr_filter[i]:
                position = 1
                signals[i] = 0.25
            # Enter short on breakout below Donchian low with volume confirmation and sufficient volatility
            elif close[i] < lowest_low[i] and volume_confirmed[i] and atr_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals