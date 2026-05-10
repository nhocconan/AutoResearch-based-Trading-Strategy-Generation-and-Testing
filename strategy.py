#!/usr/bin/env python3
# 4h_Camilla_R1_S1_Breakout_1dTrend_VolumeS
# Hypothesis: 4h price breaks Camarilla R1 or S1 levels derived from 1d OHLC, filtered by 1d EMA50 trend and volume surge. Camarilla levels provide institutional support/resistance. Works in bull (breakouts with trend) and bear (breakouts against trend filtered out). Targets 20-40 trades/year to minimize fee drag.

name = "4h_Camilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar (H, L, C)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1 = C + (H - L) * 1.12
    # Camarilla S1 = C - (H - L) * 1.12
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.12
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume average (6-period = 1.5 days of 4h bars)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # track holding period
    
    # Warmup: need EMA50 (50) + volume MA (6)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        # Determine trend from 1d EMA50
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        uptrend = close_1d_aligned[i] > ema_50_1d_aligned[i]
        downtrend = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation (1.8x average)
        volume_surge = volume[i] > 1.8 * vol_ma[i]
        
        # Camarilla breakout signals
        breakout_r1 = close[i] > camarilla_r1_aligned[i-1]
        breakdown_s1 = close[i] < camarilla_s1_aligned[i-1]
        
        if position == 0:
            bars_since_entry = 0
            # Long: Camarilla R1 breakout with volume surge and 1d uptrend
            if breakout_r1 and volume_surge and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Camarilla S1 breakdown with volume surge and 1d downtrend
            elif breakdown_s1 and volume_surge and downtrend:
                signals[i] = -0.25
                position = -1
        else:
            bars_since_entry += 1
            # Enforce minimum holding period of 3 bars (6 hours)
            if bars_since_entry < 3:
                signals[i] = signals[i-1]  # maintain position
                continue
            
            if position == 1:
                # Long exit: price breaks below Camarilla S1 or trend changes
                if close[i] < camarilla_s1_aligned[i-1] or not uptrend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: price breaks above Camarilla R1 or trend changes
                if close[i] > camarilla_r1_aligned[i-1] or not downtrend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals