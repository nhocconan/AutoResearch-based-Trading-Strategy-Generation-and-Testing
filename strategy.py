# US2314
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy combining 12h trend filter (HMA21) with 4h Donchian breakout
# Uses volume confirmation (>1.5x 20-period average) to filter breakouts
# Trend filter prevents counter-trend entries: long only when price above 12h HMA21, short only when below
# Donchian(20) breakouts capture momentum moves; exits on opposite Donchian(10) touch for quick reversion
# Works in both bull/bear markets: trend filter adapts to market direction, breakouts capture momentum
# Target: 80-180 total trades over 4 years (20-45/year) with 0.25 position sizing

name = "4h_Donchian20_HMA21_12hTrend_VolumeFilter_v1"
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
    
    # Calculate 12h HMA21 trend filter ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate HMA(21) on 12h close
    close_12h = df_12h['close'].values
    n_hma = 21
    wma2 = np.zeros_like(close_12h)
    wma1 = np.zeros_like(close_12h)
    sqrt_n = int(np.sqrt(n_hma))
    
    # WMA(n/2)
    half_n = n_hma // 2
    weights_half = np.arange(1, half_n + 1)
    sum_weights_half = weights_half.sum()
    for i in range(half_n, len(close_12h)):
        wma1[i] = np.dot(close_12h[i-half_n+1:i+1], weights_half) / sum_weights_half
    
    # WMA(n)
    weights_full = np.arange(1, n_hma + 1)
    sum_weights_full = weights_full.sum()
    for i in range(n_hma, len(close_12h)):
        wma2[i] = np.dot(close_12h[i-n_hma+1:i+1], weights_full) / sum_weights_full
    
    # HMA = 2*WMA(n/2) - WMA(n)
    hma_12h = 2 * wma1 - wma2
    # Final WMA(sqrt(n)) on the HMA
    weights_sqrt = np.arange(1, sqrt_n + 1)
    sum_weights_sqrt = weights_sqrt.sum()
    hma_final = np.zeros_like(close_12h)
    for i in range(sqrt_n, len(hma_12h)):
        hma_final[i] = np.dot(hma_12h[i-sqrt_n+1:i+1], weights_sqrt) / sum_weights_sqrt
    
    # Align 12h HMA to 4h timeframe
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_final)
    
    # Calculate 4h Donchian channels
    donchian_len = 20
    upper_donchian = pd.Series(high).rolling(window=donchian_len, min_periods=donchian_len).max().values
    lower_donchian = pd.Series(low).rolling(window=donchian_len, min_periods=donchian_len).min().values
    
    # Exit channel (shorter for reversion)
    exit_len = 10
    upper_exit = pd.Series(high).rolling(window=exit_len, min_periods=exit_len).max().values
    lower_exit = pd.Series(low).rolling(window=exit_len, min_periods=exit_len).min().values
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(hma_12h_aligned[i]) or np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or
            np.isnan(upper_exit[i]) or np.isnan(lower_exit[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above upper Donchian + volume + above 12h HMA
            if close[i] > upper_donchian[i] and volume_filter[i] and close[i] > hma_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower Donchian + volume + below 12h HMA
            elif close[i] < lower_donchian[i] and volume_filter[i] and close[i] < hma_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price touches lower exit channel (mean reversion) or breaks below Donchian
            if close[i] < lower_exit[i] or close[i] < lower_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price touches upper exit channel or breaks above Donchian
            if close[i] > upper_exit[i] or close[i] > upper_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals