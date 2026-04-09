#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ATR filter and volume spike
# In trending markets: buy breakout above 20-period high, sell breakdown below 20-period low
# Volume confirmation requires 2x average volume + ATR > 1.5x ATR(50) to ensure sufficient volatility
# Uses discrete position sizing 0.25 to limit trades to ~12-37/year and reduce fee drag
# Works in bull/bear markets: breakouts capture strong moves, ATR filter avoids low-volatility whipsaws

name = "12h_1d_donchian_breakout_atr_volume_v1"
timeframe = "12h"
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
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ATR(50) for volatility filter
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
    
    atr_1d_50 = wilders_smoothing(tr, 50)
    atr_1d_14 = wilders_smoothing(tr, 14)
    
    # Calculate 1d average volume (20-period)
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h Donchian channels (20-period) using 1d data shifted for alignment
    # We use 1d high/low to create channels, then align to 12h
    donchian_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d indicators to 12h timeframe
    atr_1d_50_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_50)
    atr_1d_14_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_14)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    donchian_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    
    # Pre-compute volume confirmation and ATR filter
    volume_confirmed = volume > 1.5 * avg_volume_1d_aligned
    atr_filter = atr_1d_14_aligned > (1.2 * atr_1d_50_aligned)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_1d_aligned[i]) or np.isnan(donchian_low_1d_aligned[i]) or
            np.isnan(atr_filter[i]) or np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long if price breaks below Donchian low or volatility drops
            if close[i] < donchian_low_1d_aligned[i] or not atr_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price breaks above Donchian high or volatility drops
            if close[i] > donchian_high_1d_aligned[i] or not atr_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long on breakout above Donchian high with volume and ATR confirmation
            if close[i] > donchian_high_1d_aligned[i] and volume_confirmed[i] and atr_filter[i]:
                position = 1
                signals[i] = 0.25
            # Enter short on breakout below Donchian low with volume and ATR confirmation
            elif close[i] < donchian_low_1d_aligned[i] and volume_confirmed[i] and atr_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals