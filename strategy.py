#!/usr/bin/env python3
# 6h_12h_volume_vwap_breakout_v1
# Hypothesis: 6-hour VWAP breakout with 12-hour volume confirmation and volume-weighted RSI filter.
# Long when price breaks above VWAP with rising volume and VWAP-RSI < 30.
# Short when price breaks below VWAP with rising volume and VWAP-RSI > 70.
# Uses VWAP on 6h for entry timing and 12h volume for confirmation to avoid fakeouts.
# Designed to generate ~20-40 trades/year to avoid fee decay while capturing momentum bursts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_volume_vwap_breakout_v1"
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
    
    # Calculate VWAP on 6h data
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = np.where(vwap_denominator != 0, vwap_numerator / vwap_denominator, np.nan)
    
    # Calculate volume-weighted RSI on 6h data
    price_change = np.diff(close, prepend=close[0])
    gain = np.where(price_change > 0, price_change, 0.0)
    loss = np.where(price_change < 0, -price_change, 0.0)
    
    # Volume-weighted gain and loss
    vgain = gain * volume
    vloss = loss * volume
    
    # Wilder's smoothing with volume weighting
    avg_vgain = np.zeros(n)
    avg_vloss = np.zeros(n)
    avg_vgain[0] = vgain[0]
    avg_vloss[0] = vloss[0]
    
    for i in range(1, n):
        avg_vgain[i] = (avg_vgain[i-1] * 13 + vgain[i]) / 14
        avg_vloss[i] = (avg_vloss[i-1] * 13 + vloss[i]) / 14
    
    rs = np.where(avg_vloss != 0, avg_vgain / avg_vloss, 100)
    vwap_rsi = 100 - (100 / (1 + rs))
    
    # Get 12-hour data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    
    # Calculate volume moving average and ratio on 12h
    vol_ma_12h = np.zeros(len(volume_12h))
    for i in range(len(volume_12h)):
        if i < 19:
            vol_ma_12h[i] = np.nan
        else:
            vol_ma_12h[i] = np.mean(volume_12h[i-19:i+1])
    
    vol_ratio_12h = np.where(vol_ma_12h != 0, volume_12h / vol_ma_12h, np.nan)
    
    # Align 12h volume ratio to 6h timeframe
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(vwap[i]) or np.isnan(vwap_rsi[i]) or 
            np.isnan(vol_ratio_12h_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        vwap_val = vwap[i]
        vwap_rsi_val = vwap_rsi[i]
        vol_ratio = vol_ratio_12h_aligned[i]
        
        if position == 1:  # Long
            # Exit: price breaks below VWAP or VWAP-RSI > 70
            if price < vwap_val or vwap_rsi_val > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price breaks above VWAP or VWAP-RSI < 30
            if price > vwap_val or vwap_rsi_val < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry conditions: VWAP breakout with volume confirmation
            # Bullish: price breaks above VWAP, volume above average, and VWAP-RSI < 30
            if price > vwap_val and vol_ratio > 1.5 and vwap_rsi_val < 30:
                position = 1
                signals[i] = 0.25
            # Bearish: price breaks below VWAP, volume above average, and VWAP-RSI > 70
            elif price < vwap_val and vol_ratio > 1.5 and vwap_rsi_val > 70:
                position = -1
                signals[i] = -0.25
    
    return signals