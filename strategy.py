#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend
Hypothesis: Price breaks 1-hourly Camarilla R1 (long) or S1 (short) levels calculated from prior day's range, with 4h EMA50 trend filter and volume confirmation.
Uses 4h for trend direction and 1h for precise entry timing to reduce noise and maintain low trade frequency.
Target: 15-35 trades/year (60-140 total) to minimize fee drag while capturing directional moves in both bull and bear markets.
"""

name = "1h_Camarilla_R1_S1_Breakout_4hTrend"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h data for trend and regime
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h EMA50 for trend filter
    ema50_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 50:
        ema50_4h[49] = np.mean(close_4h[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_4h)):
            ema50_4h[i] = alpha * close_4h[i] + (1 - alpha) * ema50_4h[i-1]
    
    # Calculate 4h volume SMA20 for volume confirmation
    vol_sma20_4h = np.full(len(volume_4h), np.nan)
    if len(volume_4h) >= 20:
        vol_sma20_4h[19] = np.mean(volume_4h[:20])
        for i in range(20, len(volume_4h)):
            vol_sma20_4h[i] = (vol_sma20_4h[i-1] * 19 + volume_4h[i]) / 20
    
    # Calculate daily Camarilla levels from prior day
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align 4h indicators to 1h
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    vol_sma20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_sma20_4h)
    
    # Align daily Camarilla levels to 1h
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate session filter (08:00-20:00 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50
    
    for i in range(start_idx, n):
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_sma20_4h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 1h volume > 1.5x average 4h volume (scaled)
        vol_4h_scaled = vol_sma20_4h_aligned[i] / 4.0  # 4x 1h bars in 4h
        volume_confirm = volume[i] > 1.5 * vol_4h_scaled
        
        # Trend and price relative to Camarilla levels
        is_uptrend = close[i] > ema50_4h_aligned[i]
        is_downtrend = close[i] < ema50_4h_aligned[i]
        price_above_r1 = close[i] > r1_aligned[i]
        price_below_s1 = close[i] < s1_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1, in uptrend, with volume
            if price_above_r1 and is_uptrend and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1, in downtrend, with volume
            elif price_below_s1 and is_downtrend and volume_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price falls back below R1 or trend turns down
            if not price_above_r1 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price rises back above S1 or trend turns up
            if not price_below_s1 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals