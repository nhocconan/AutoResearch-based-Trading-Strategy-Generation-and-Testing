#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for VWAP reference
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12-hour VWAP (typical price * volume / cumulative volume)
    typical_price_12h = (high_12h + low_12h + close_12h) / 3
    cum_vol = np.cumsum(volume_12h)
    cum_tpv = np.cumsum(typical_price_12h * volume_12h)
    vwap_12h = np.where(cum_vol > 0, cum_tpv / cum_vol, np.nan)
    
    # Align 12h VWAP to primary timeframe
    vwap_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h)
    
    # Calculate 12h ATR for volatility filter
    if len(high_12h) < 14:
        return np.zeros(n)
    
    tr = np.zeros_like(high_12h)
    for i in range(1, len(high_12h)):
        if np.isnan(high_12h[i]) or np.isnan(low_12h[i]) or np.isnan(high_12h[i-1]) or np.isnan(low_12h[i-1]):
            continue
        tr[i] = max(high_12h[i] - low_12h[i], 
                   abs(high_12h[i] - high_12h[i-1]), 
                   abs(low_12h[i] - low_12h[i-1]))
    
    atr_12h = np.full_like(high_12h, np.nan)
    if len(high_12h) >= 14:
        atr_12h[13] = np.nanmean(tr[1:14])
        for i in range(14, len(high_12h)):
            if np.isnan(tr[i]):
                atr_12h[i] = atr_12h[i-1]
            else:
                atr_12h[i] = (atr_12h[i-1] * 13 + tr[i]) / 14
    
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # Conservative size to limit trades
    
    for i in range(14, n):
        # Skip if any critical data is NaN
        if np.isnan(vwap_12h_aligned[i]) or np.isnan(atr_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current period volume vs 20-period average
        vol_ma_20 = np.full_like(volume, np.nan)
        for j in range(19, len(volume)):
            vol_ma_20[j] = np.mean(volume[j-19:j+1])
        
        if np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        if position == 0:
            # Long: price below VWAP with volume surge (mean reversion in range)
            if (close[i] < vwap_12h_aligned[i] and 
                volume_ratio > 2.0):
                position = 1
                signals[i] = position_size
            # Short: price above VWAP with volume surge (mean reversion in range)
            elif (close[i] > vwap_12h_aligned[i] and 
                  volume_ratio > 2.0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses above VWAP or volume dries up
            if (close[i] > vwap_12h_aligned[i] or
                volume_ratio < 0.5):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses below VWAP or volume dries up
            if (close[i] < vwap_12h_aligned[i] or
                volume_ratio < 0.5):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_12h_VWAP_MeanReversion_Volume"
timeframe = "12h"
leverage = 1.0