#!/usr/bin/env python3
"""
6h Camarilla Pivot Breakout with 1d EMA34 Trend Filter and Volume Spike
Hypothesis: Camarilla R3/S3 levels act as strong intraday support/resistance. 
Breakouts above R3 or below S3 with volume confirmation and aligned 1d EMA34 trend 
yield high-probability trades. Works in bull/bear via trend filter.
Target: 12-37 trades/year (50-150 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla_pivots(high, low, close):
    """
    Calculate Camarilla pivot levels for given high, low, close.
    Returns (R4, R3, R2, R1, PP, S1, S2, S3, S4)
    """
    range_ = high - low
    if range_ == 0:
        return close, close, close, close, close, close, close, close, close
    pp = (high + low + close) / 3.0
    r1 = close + range_ * 1.1 / 12
    r2 = close + range_ * 1.1 / 6
    r3 = close + range_ * 1.1 / 4
    r4 = close + range_ * 1.1 / 2
    s1 = close - range_ * 1.1 / 12
    s2 = close - range_ * 1.1 / 6
    s3 = close - range_ * 1.1 / 4
    s4 = close - range_ * 1.1 / 2
    return r4, r3, r2, r1, pp, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivots from 1d OHLC
    r4_1d, r3_1d, r2_1d, r1_1d, pp_1d, s1_1d, s2_1d, s3_1d, s4_1d = calculate_camarilla_pivots(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    # Align Camarilla levels to 6h timeframe (no additional delay needed for pivots)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA34 warmup
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_34_aligned[i]) or np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_34_aligned[i]
        r3_level = r3_1d_aligned[i]
        s3_level = s3_1d_aligned[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Breakout signals with trend filter
        if position == 0:
            # Long: price breaks above R3 with volume spike AND above 1d EMA34 (uptrend)
            long_condition = (curr_close > r3_level) and volume_spike and (curr_close > ema_trend)
            # Short: price breaks below S3 with volume spike AND below 1d EMA34 (downtrend)
            short_condition = (curr_close < s3_level) and volume_spike and (curr_close < ema_trend)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below R3 or trend breaks
            if curr_close < r3_level or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above S3 or trend breaks
            if curr_close > s3_level or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0