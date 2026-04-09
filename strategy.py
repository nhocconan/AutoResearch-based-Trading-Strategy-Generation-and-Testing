#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and ATR(14) volatility filter
# In bull/bear markets: Donchian breakout catches strong trends, volume confirmation avoids false breakouts
# ATR filter ensures sufficient volatility for meaningful moves. Uses discrete sizing 0.25 to limit trades to ~20-50/year.
# Works in ranging markets by reducing position size when ATR low (implicit via volatility filter)

name = "4h_1d_donchian_volume_atr_v1"
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
    
    # Volume spike: current 1d volume > 2.0 * 20-period average
    volume_spike = np.where(volume_1d > 2.0 * avg_volume_1d, 1.0, 0.0)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # ATR filter: 1d ATR > 20-period average ATR (avoid low volatility periods)
    atr_s_1d = pd.Series(atr_1d)
    avg_atr_1d = atr_s_1d.rolling(window=20, min_periods=20).mean().values
    atr_filter = np.where(atr_1d > avg_atr_1d, 1.0, 0.0)
    atr_filter_aligned = align_htf_to_ltf(prices, df_1d, atr_filter)
    
    # Calculate 4h Donchian channels (20-period)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().shift(1).values  # shift(1) to avoid look-ahead
    donchian_low = low_s.rolling(window=20, min_periods=20).min().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(volume_spike_aligned[i]) or np.isnan(atr_filter_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        # Volume and volatility filters must both be present
        if volume_spike_aligned[i] < 1.0 or atr_filter_aligned[i] < 1.0:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long if price breaks below Donchian low
            if close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price breaks above Donchian high
            if close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long on breakout above Donchian high
            if close[i] > donchian_high[i]:
                position = 1
                signals[i] = 0.25
            # Enter short on breakout below Donchian low
            elif close[i] < donchian_low[i]:
                position = -1
                signals[i] = -0.25
    
    return signals