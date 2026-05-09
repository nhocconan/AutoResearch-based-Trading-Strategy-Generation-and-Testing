#!/usr/bin/env python3
name = "4H_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
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
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla pivot levels from previous day
    camarilla_h1 = np.full_like(close_1d, np.nan)
    camarilla_l1 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_ = prev_high - prev_low
        camarilla_h1[i] = prev_close + 1.1 * range_ / 12
        camarilla_l1[i] = prev_close - 1.1 * range_ / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    
    # Get daily close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Get 1h data for volume confirmation
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)
    
    volume_1h = df_1h['volume'].values
    
    # Calculate 1h volume EMA20
    vol_ema20_1h = pd.Series(volume_1h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1h volume EMA20 to 4h timeframe
    vol_ema20_1h_aligned = align_htf_to_ltf(prices, df_1h, vol_ema20_1h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h1_aligned[i]) or np.isnan(camarilla_l1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ema20_1h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market conditions
        # Uptrend: price above daily EMA34
        uptrend = close[i] > ema34_1d_aligned[i]
        # Downtrend: price below daily EMA34
        downtrend = close[i] < ema34_1d_aligned[i]
        # Volume surge: current volume > 2.0x 1h volume EMA20
        volume_surge = volume[i] > vol_ema20_1h_aligned[i] * 2.0
        
        if position == 0:
            # Enter long: Uptrend + price breaks above Camarilla H1 + volume surge
            if uptrend and close[i] > camarilla_h1_aligned[i] and volume_surge:
                signals[i] = 0.25
                position = 1
            # Enter short: Downtrend + price breaks below Camarilla L1 + volume surge
            elif downtrend and close[i] < camarilla_l1_aligned[i] and volume_surge:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Trend turns down OR price breaks below Camarilla L1
            if not uptrend or close[i] < camarilla_l1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Trend turns up OR price breaks above Camarilla H1
            if not downtrend or close[i] > camarilla_h1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals