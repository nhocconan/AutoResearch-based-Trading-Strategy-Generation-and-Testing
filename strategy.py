#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1w ATR filter and 1d volume confirmation
# In trending markets: breakout above/below 20-period Donchian channels with volume confirmation
# Uses 1w ATR to filter low volatility environments and 1d volume spike for confirmation
# Discrete position sizing 0.25 to limit trades to ~12-37/year and reduce fee drag
# Works in bull/bear markets: breakout catches strong moves, ATR filter avoids choppy markets

name = "12h_1w_donchian_breakout_volume_v1"
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
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ATR(20) for volatility filter
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
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
    
    atr_1w = wilders_smoothing(tr, 20)
    atr_ma_1w = pd.Series(atr_1w).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d average volume (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align HTF indicators to 12h timeframe
    atr_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_ma_1w)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # Pre-compute signals
    volume_confirmed = volume > 1.5 * avg_volume_1d_aligned
    volatility_filter = atr_ma_1w_aligned > 0  # Only trade when volatility is sufficient
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(volume_confirmed[i]) or np.isnan(volatility_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long if price breaks below Donchian low or volatility drops
            if close[i] < lowest_low[i] or not volatility_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit short if price breaks above Donchian high or volatility drops
            if close[i] > highest_high[i] or not volatility_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long on breakout above Donchian high with volume confirmation and sufficient volatility
            if close[i] > highest_high[i] and volume_confirmed[i] and volatility_filter[i]:
                position = 1
                signals[i] = 0.25
            # Enter short on breakout below Donchian low with volume confirmation and sufficient volatility
            elif close[i] < lowest_low[i] and volume_confirmed[i] and volatility_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals