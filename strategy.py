#!/usr/bin/env python3
# 6h_WeeklyVWAP_Pullback_With_Volume
# Hypothesis: Pullbacks to weekly VWAP during strong trends with volume confirmation.
# Long in uptrend when price pulls back to weekly VWAP with rising volume.
# Short in downtrend when price rallies to weekly VWAP with rising volume.
# Weekly trend determined by price above/below weekly VWAP (institutional fair value).
# Volume confirmation requires current volume > 1.5x 20-period average to avoid low-quality signals.
# Target: 15-30 trades/year per symbol with disciplined risk management in both bull and bear markets.

name = "6h_WeeklyVWAP_Pullback_With_Volume"
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
    
    # Get weekly data for VWAP and trend
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    volume_weekly = df_weekly['volume'].values
    
    # Calculate weekly VWAP (volume-weighted average price)
    typical_price_weekly = (high_weekly + low_weekly + close_weekly) / 3.0
    vp_weekly = typical_price_weekly * volume_weekly
    
    # Cumulative values for VWAP calculation
    cum_vp = np.cumsum(vp_weekly)
    cum_volume = np.cumsum(volume_weekly)
    vwap_weekly = np.divide(cum_vp, cum_volume, out=np.full_like(cum_vp, np.nan), where=cum_volume!=0)
    
    # Weekly trend: price above/below VWAP
    vwap_weekly_aligned = align_htf_to_ltf(prices, df_weekly, vwap_weekly)
    
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
    
    start_idx = max(20, 1)  # Need volume MA and weekly VWAP
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vwap_weekly_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend based on price vs VWAP
        weekly_up = close[i] > vwap_weekly_aligned[i]
        
        if position == 0:
            # Enter long: weekly uptrend + price pulls back to VWAP + volume confirmation
            if weekly_up and close[i] <= vwap_weekly_aligned[i] * 1.005 and volume_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly downtrend + price rallies to VWAP + volume confirmation
            elif not weekly_up and close[i] >= vwap_weekly_aligned[i] * 0.995 and volume_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: weekly trend turns down or price moves significantly above VWAP
            if not weekly_up or close[i] > vwap_weekly_aligned[i] * 1.02:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: weekly trend turns up or price moves significantly below VWAP
            if weekly_up or close[i] < vwap_weekly_aligned[i] * 0.98:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals