#!/usr/bin/env python3
# 6h_1d_price_channel_breakout_v2
# Hypothesis: 6-hour breakout of daily Donchian channels with daily volatility filter and volume confirmation.
# Long when price breaks above daily Donchian upper band with volatility > 1.5x 50-period average and volume > 1.5x 20-bar average.
# Short when price breaks below daily Donchian lower band with volatility > 1.5x 50-period average and volume > 1.5x 20-bar average.
# Exit when price returns to opposite Donchian band or volatility drops below threshold.
# Works in trending markets via breakout continuation and in ranging markets via mean reversion at channel extremes.
# Uses volatility filter to avoid false breakouts in low volatility environments.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_price_channel_breakout_v2"
timeframe = "6h"
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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = np.full(len(df_1d), np.nan)
    donchian_low = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if i >= 19:
            donchian_high[i] = np.max(high_1d[i-19:i+1])
            donchian_low[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate daily ATR (14-period) for volatility filter
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]]))
    low_close = np.abs(low_1d - np.concatenate([[low_1d[0]], low_1d[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14 = np.full(len(tr), np.nan)
    
    for i in range(len(tr)):
        if i < 14:
            atr_14[i] = np.nan
        elif i == 14:
            atr_14[i] = np.mean(tr[0:15])
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Calculate 50-period average ATR for volatility threshold
    atr_ma_50 = np.full(len(atr_14), np.nan)
    atr_sum = 0
    for i in range(len(atr_14)):
        if not np.isnan(atr_14[i]):
            atr_sum += atr_14[i]
            if i >= 50:
                atr_sum -= atr_14[i-50]
            if i >= 49:
                atr_ma_50[i] = atr_sum / 50
    
    # Align indicators to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    atr_ma_50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50)
    
    # Volume confirmation: 20-period average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr_ma_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to or below lower band OR volatility drops below threshold
            if (close[i] <= donchian_low_aligned[i] or 
                atr_ma_50_aligned[i] < np.mean(atr_ma_50_aligned[max(0, i-50):i]) * 0.5):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to or above upper band OR volatility drops below threshold
            if (close[i] >= donchian_high_aligned[i] or 
                atr_ma_50_aligned[i] < np.mean(atr_ma_50_aligned[max(0, i-50):i]) * 0.5):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Calculate current volatility relative to 50-period average
            current_vol_ratio = atr_ma_50_aligned[i] / np.mean(atr_ma_50_aligned[max(0, i-50):i]) if i >= 50 else 1.0
            
            # Enter long: price breaks above upper band with volatility and volume filters
            if (close[i] > donchian_high_aligned[i] and 
                current_vol_ratio > 1.5 and 
                volume[i] > vol_ma_20[i] * 1.5):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below lower band with volatility and volume filters
            elif (close[i] < donchian_low_aligned[i] and 
                  current_vol_ratio > 1.5 and 
                  volume[i] > vol_ma_20[i] * 1.5):
                position = -1
                signals[i] = -0.25
    
    return signals