# 12h_4hTrend_1dVolume_Spike_Regime
# Hypothesis: On 12h timeframe, use 4h Supertrend for trend direction, 1d volume spike for momentum confirmation, and 12h Choppiness Index to filter range-bound markets. This combination aims to capture strong trending moves while avoiding whipsaws in sideways markets, working in both bull and bear regimes by following the 4h trend.
# Target: 50-150 total trades over 4 years (12-37/year) with selective entries.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Supertrend trend filter
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h ATR(10)
    tr_4h = np.maximum(high_4h[1:] - low_4h[1:], np.abs(high_4h[1:] - close_4h[:-1]), np.abs(low_4h[1:] - close_4h[:-1]))
    tr_4h = np.concatenate([[np.nan], tr_4h])
    atr_10_4h = pd.Series(tr_4h).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 4h Supertrend (10, 3.0)
    hl2_4h = (high_4h + low_4h) / 2
    upper_band_4h = hl2_4h + 3.0 * atr_10_4h
    lower_band_4h = hl2_4h - 3.0 * atr_10_4h
    
    supertrend_4h = np.full_like(close_4h, np.nan)
    dir_4h = np.ones_like(close_4h)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_4h)):
        if np.isnan(atr_10_4h[i]):
            supertrend_4h[i] = np.nan
            continue
        if close_4h[i-1] > upper_band_4h[i-1]:
            dir_4h[i] = 1
        elif close_4h[i-1] < lower_band_4h[i-1]:
            dir_4h[i] = -1
        else:
            dir_4h[i] = dir_4h[i-1]
            if dir_4h[i] == 1 and lower_band_4h[i] < lower_band_4h[i-1]:
                lower_band_4h[i] = lower_band_4h[i-1]
            if dir_4h[i] == -1 and upper_band_4h[i] > upper_band_4h[i-1]:
                upper_band_4h[i] = upper_band_4h[i-1]
        supertrend_4h[i] = lower_band_4h[i] if dir_4h[i] == 1 else upper_band_4h[i]
    
    supertrend_4h_aligned = align_htf_to_ltf(prices, df_4h, supertrend_4h)
    
    # Get 1d data for volume spike
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume SMA(20)
    vol_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    # Get 12h data for Choppiness Index
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ATR(14)
    tr_12h = np.maximum(high_12h[1:] - low_12h[1:], np.abs(high_12h[1:] - close_12h[:-1]), np.abs(low_12h[1:] - close_12h[:-1]))
    tr_12h = np.concatenate([[np.nan], tr_12h])
    atr_14_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12h True Range sum and high-low range
    tr_sum_14 = pd.Series(tr_12h).rolling(window=14, min_periods=14).sum().values
    hh_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    range_14[range_14 == 0] = np.nan
    
    # Calculate 12h Choppiness Index (14)
    chop_14 = 100 * np.log10(tr_sum_14 / range_14) / np.log10(14)
    chop_14_aligned = align_htf_to_ltf(prices, df_12h, chop_14)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(30, 50)  # ensure indicators are ready
    
    for i in range(start_idx, n):
        if (np.isnan(supertrend_4h_aligned[i]) or np.isnan(vol_sma_20_1d_aligned[i]) or
            np.isnan(chop_14_aligned[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_20_1d_aligned[i]
        st_4h = supertrend_4h_aligned[i]
        chop = chop_14_aligned[i]
        
        if position == 0:
            # Long: 4h Supertrend uptrend + price above Supertrend + volume spike + chop < 61.8 (trending)
            if price > st_4h and vol > 2.5 * vol_sma_val and chop < 61.8:
                signals[i] = 0.25
                position = 1
            # Short: 4h Supertrend downtrend + price below Supertrend + volume spike + chop < 61.8 (trending)
            elif price < st_4h and vol > 2.5 * vol_sma_val and chop < 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Supertrend or chop > 61.8 (range) or volume drops
            if price < st_4h or chop > 61.8 or vol < 0.5 * vol_sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Supertrend or chop > 61.8 (range) or volume drops
            if price > st_4h or chop > 61.8 or vol < 0.5 * vol_sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_4hTrend_1dVolume_Spike_Regime"
timeframe = "12h"
leverage = 1.0