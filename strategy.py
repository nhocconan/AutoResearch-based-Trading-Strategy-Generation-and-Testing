#!/usr/bin/env python3
"""
1d_WVWAP_Cross_Volume_Surge
Hypothesis: 1d volume-weighted average price (VWAP) cross with volume surge acts as institutional order flow signal. Works in both bull/bear markets as VWAP captures fair value and volume surge confirms participation. Uses 1w trend filter to avoid counter-trend trades. Target: 15-25 trades/year.
"""

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
    
    # Calculate 1d VWAP (typical price * volume cumulative / volume cumulative)
    df_1d = get_htf_data(prices, '1d')
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    pv_1d = typical_price_1d * df_1d['volume'].values
    vol_1d = df_1d['volume'].values
    
    # Cumulative VWAP calculation
    cum_pv = np.cumsum(pv_1d)
    cum_vol = np.cumsum(vol_1d)
    vwap_1d = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol!=0)
    
    # 1w EMA50 trend filter (avoid counter-trend in chop)
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 1d timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume surge: current 1d volume > 2.5x 20-day average
    vol_ma_20 = np.full(len(close), np.nan)
    for i in range(20, len(close)):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_surge = volume > (vol_ma_20 * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above VWAP with volume surge and 1w uptrend
            if (close[i] > vwap_1d_aligned[i] and volume_surge[i] and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below VWAP with volume surge and 1w downtrend
            elif (close[i] < vwap_1d_aligned[i] and volume_surge[i] and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below VWAP or 1w trend turns down
            if (close[i] < vwap_1d_aligned[i] or close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above VWAP or 1w trend turns up
            if (close[i] > vwap_1d_aligned[i] or close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WVWAP_Cross_Volume_Surge"
timeframe = "1d"
leverage = 1.0