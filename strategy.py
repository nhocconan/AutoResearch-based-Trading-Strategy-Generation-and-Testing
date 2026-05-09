#!/usr/bin/env python3
# 1d_WVWAP_Volume_Spike_Trend
# Strategy: Weighted VWAP deviation with volume spike confirmation and weekly trend filter
# Long when price > VWAP(20) + volume > 1.5x avg volume and weekly close > weekly open
# Short when price < VWAP(20) - volume > 1.5x avg volume and weekly close < weekly open
# Exit when price crosses VWAP(20)
# Designed for 1d timeframe to capture mean reversion with institutional volume confirmation
# Uses weekly trend filter to avoid counter-trend trades in strong trends

name = "1d_WVWAP_Volume_Spike_Trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate VWAP(20)
    typical_price = (high + low + close) / 3.0
    vp = typical_price * volume
    
    # Calculate cumulative sums for VWAP
    cum_vp = np.zeros(n)
    cum_vol = np.zeros(n)
    
    cum_vp[0] = vp[0]
    cum_vol[0] = volume[0]
    
    for i in range(1, n):
        cum_vp[i] = cum_vp[i-1] + vp[i]
        cum_vol[i] = cum_vol[i-1] + volume[i]
    
    vwap = np.full(n, np.nan)
    for i in range(20, n):  # 20-period VWAP
        start_idx = i - 19
        vp_sum = cum_vp[i] - (cum_vp[start_idx-1] if start_idx > 0 else 0)
        vol_sum = cum_vol[i] - (cum_vol[start_idx-1] if start_idx > 0 else 0)
        if vol_sum > 0:
            vwap[i] = vp_sum / vol_sum
    
    # Calculate average volume (20-period)
    avg_vol = np.full(n, np.nan)
    for i in range(20, n):
        start_idx = i - 19
        vol_sum = cum_vol[i] - (cum_vol[start_idx-1] if start_idx > 0 else 0)
        avg_vol[i] = vol_sum / 20.0
    
    # Volume spike condition: volume > 1.5x average volume
    volume_spike = volume > (1.5 * avg_vol)
    
    # Price deviation from VWAP
    vwap_dev = close - vwap
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly trend: bullish if weekly close > weekly open
    weekly_bullish = df_1w['close'] > df_1w['open']
    weekly_bullish_vals = weekly_bullish.values
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish_vals)
    
    # Weekly bearish if weekly close < weekly open
    weekly_bearish = df_1w['close'] < df_1w['open']
    weekly_bearish_vals = weekly_bearish.values
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish_vals)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for VWAP
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vwap[i]) or np.isnan(avg_vol[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above VWAP, volume spike, and weekly bullish
            if (vwap_dev[i] > 0 and volume_spike[i] and 
                weekly_bullish_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price below VWAP, volume spike, and weekly bearish
            elif (vwap_dev[i] < 0 and volume_spike[i] and 
                  weekly_bearish_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below VWAP
            if vwap_dev[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above VWAP
            if vwap_dev[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals