#!/usr/bin/env python3
"""
4h_VWAP_MeanReversion_TrendFilter
Hypothesis: Mean reversion to daily VWAP with 1-week EMA200 trend filter and volume confirmation.
Long when price is below VWAP in uptrend (price > weekly EMA200), short when above VWAP in downtrend.
Volume spike (>1.8x 48-period average) confirms mean reversion strength. Designed for low trade frequency.
"""

name = "4h_VWAP_MeanReversion_TrendFilter"
timeframe = "4h"
leverage = 1.0

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
    
    # Get daily data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily VWAP (typical price * volume) / cumulative volume
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    pv_1d = typical_price_1d * volume_1d
    cum_pv_1d = np.cumsum(pv_1d)
    cum_vol_1d = np.cumsum(volume_1d)
    vwap_1d = np.divide(cum_pv_1d, cum_vol_1d, out=np.full_like(cum_pv_1d, np.nan), where=cum_vol_1d!=0)
    
    # Align daily VWAP to 4h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA200 for trend filter
    ema_200_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 200:
        ema_200_1w[199] = np.mean(close_1w[0:200])
        for i in range(200, len(close_1w)):
            ema_200_1w[i] = (ema_200_1w[i-1] * 199 + close_1w[i]) / 200
    
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume spike filter: current volume / 48-period average volume (48*4h = 8 days)
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 48:
        vol_ma[47] = np.mean(volume[0:48])
        for i in range(48, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 47 + volume[i]) / 48
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(48, 200)  # Ensure volume MA and weekly EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Enter long: price below VWAP AND uptrend (price > weekly EMA200) AND volume spike
            if (close[i] < vwap_1d_aligned[i] and 
                close[i] > ema_200_1w_aligned[i] and 
                volume_ratio[i] > 1.8):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Enter short: price above VWAP AND downtrend (price < weekly EMA200) AND volume spike
            elif (close[i] > vwap_1d_aligned[i] and 
                  close[i] < ema_200_1w_aligned[i] and 
                  volume_ratio[i] > 1.8):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Minimum holding period: 4 bars
            if bars_since_entry < 4:
                signals[i] = 0.25
            else:
                # Exit long: price crosses above VWAP OR trend reversal (price < weekly EMA200)
                if close[i] > vwap_1d_aligned[i] or close[i] < ema_200_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Minimum holding period: 4 bars
            if bars_since_entry < 4:
                signals[i] = -0.25
            else:
                # Exit short: price crosses below VWAP OR trend reversal (price > weekly EMA200)
                if close[i] < vwap_1d_aligned[i] or close[i] > ema_200_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals