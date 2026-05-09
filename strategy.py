#!/usr/bin/env python3
"""
12h_VWAP_Trend_MeanReversion_1dVwapCross
Hypothesis: Mean-reversion trades around daily VWAP with 1d trend filter and volume spike confirmation.
In trending markets (price above/below daily VWAP with 1d EMA), we take pullbacks to VWAP.
In ranging markets, we fade VWAP extremes with volume confirmation.
Designed for low trade frequency (12-37/year) to minimize fee drift. Works in both bull and bear markets.
"""

name = "12h_VWAP_Trend_MeanReversion_1dVwapCross"
timeframe = "12h"
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
    
    # Get daily data for VWAP and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily VWAP (typical price * volume)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vp_1d = typical_price_1d * volume_1d
    
    # Cumulative VWAP calculation (reset daily)
    cum_vp_1d = np.full_like(vp_1d, np.nan)
    cum_vol_1d = np.full_like(volume_1d, np.nan)
    
    for i in range(len(vp_1d)):
        if i == 0:
            cum_vp_1d[i] = vp_1d[i]
            cum_vol_1d[i] = volume_1d[i]
        else:
            cum_vp_1d[i] = cum_vp_1d[i-1] + vp_1d[i]
            cum_vol_1d[i] = cum_vol_1d[i-1] + volume_1d[i]
    
    vwap_1d = np.full_like(close_1d, np.nan)
    valid_vol = cum_vol_1d != 0
    vwap_1d[valid_vol] = cum_vp_1d[valid_vol] / cum_vol_1d[valid_vol]
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (ema_50_1d[i-1] * 49 + close_1d[i]) / 50
    
    # Align daily VWAP and EMA50 to 12h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike filter: current volume / 24-period average volume (24*12h = 12 days)
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 24:
        vol_ma[23] = np.mean(volume[0:24])
        for i in range(24, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 23 + volume[i]) / 24
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(24, 50)  # Ensure volume MA and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Enter long: price pulls back to VWAP from above in uptrend OR fades VWAP extreme in range
            # Long condition 1: Uptrend pullback (price > EMA50 and price <= VWAP)
            if (close[i] > ema_50_1d_aligned[i] and close[i] <= vwap_1d_aligned[i] and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Long condition 2: Fade VWAP extreme (price significantly below VWAP with volume)
            elif (close[i] < vwap_1d_aligned[i] * 0.98 and  # 2% below VWAP
                  volume_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Enter short: price pulls back to VWAP from below in downtrend OR fades VWAP extreme in range
            # Short condition 1: Downtrend pullback (price < EMA50 and price >= VWAP)
            elif (close[i] < ema_50_1d_aligned[i] and close[i] >= vwap_1d_aligned[i] and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
            # Short condition 2: Fade VWAP extreme (price significantly above VWAP with volume)
            elif (close[i] > vwap_1d_aligned[i] * 1.02 and  # 2% above VWAP
                  volume_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Minimum holding period: 2 bars
            if bars_since_entry < 2:
                signals[i] = 0.25
            else:
                # Exit long: price moves above VWAP (for trend trades) or mean reversion complete
                if close[i] >= vwap_1d_aligned[i] or bars_since_entry >= 8:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Minimum holding period: 2 bars
            if bars_since_entry < 2:
                signals[i] = -0.25
            else:
                # Exit short: price moves below VWAP (for trend trades) or mean reversion complete
                if close[i] <= vwap_1d_aligned[i] or bars_since_entry >= 8:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals