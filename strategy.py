#!/usr/bin/env python3
"""
6h_OrderFlow_Imbalance_Pullback_v2
Hypothesis: 6h price pullbacks to institutional order flow zones (identified by volume-weighted average price)
combined with 1d trend alignment and volume confirmation yield high-probability entries.
Works in bull/bear by trading with higher timeframe trend only.
Target: 20-50 trades/year via strict confluence of VWAP deviation, 1d EMA trend, and volume surge.
"""

name = "6h_OrderFlow_Imbalance_Pullback_v2"
timeframe = "6h"
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
    
    # Get 1d data for trend filter and VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema50_1d[i] = (close_1d[i] * 2 + ema50_1d[i-1] * 48) / 50
    
    # Align 1d EMA50 to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 6-period VWAP (typical price * volume) / volume
    typical_price = (high + low + close) / 3.0
    vp = typical_price * volume
    
    vwap_numerator = np.full_like(vp, np.nan)
    vwap_denominator = np.full_like(volume, np.nan)
    
    if len(vp) >= 6:
        vwap_numerator[5] = np.sum(vp[0:6])
        vwap_denominator[5] = np.sum(volume[0:6])
        for i in range(6, len(vp)):
            vwap_numerator[i] = vwap_numerator[i-1] + vp[i] - vp[i-6]
            vwap_denominator[i] = vwap_denominator[i-1] + volume[i] - volume[i-6]
    
    vwap = np.full_like(close, np.nan)
    valid_denom = vwap_denominator != 0
    vwap[valid_denom] = vwap_numerator[valid_denom] / vwap_denominator[valid_denom]
    
    # Calculate VWAP deviation as percentage
    vwap_dev_pct = np.full_like(close, np.nan)
    valid_vwap = ~np.isnan(vwap)
    vwap_dev_pct[valid_vwap] = (close[valid_vwap] - vwap[valid_vwap]) / vwap[valid_vwap] * 100.0
    
    # Volume filter: current volume vs 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 6, 20)  # Need 1d EMA50, VWAP(6), and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(vwap_dev_pct[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market conditions
        trend_up = close[i] > ema50_1d_aligned[i]
        vwap_dev = vwap_dev_pct[i]
        volume_surge = volume_ratio[i] > 1.5
        
        if position == 0:
            # Enter long: Uptrend + price below VWAP (pullback) + volume surge
            if trend_up and vwap_dev < -0.5 and volume_surge:  # 0.5% below VWAP
                signals[i] = 0.25
                position = 1
            # Enter short: Downtrend + price above VWAP (pullback) + volume surge
            elif not trend_up and vwap_dev > 0.5 and volume_surge:  # 0.5% above VWAP
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Trend turns down OR price crosses above VWAP OR volume dries up
            if not trend_up or vwap_dev > 0.2 or volume_ratio[i] < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Trend turns up OR price crosses below VWAP OR volume dries up
            if trend_up or vwap_dev < -0.2 or volume_ratio[i] < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals