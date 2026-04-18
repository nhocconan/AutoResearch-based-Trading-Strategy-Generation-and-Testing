#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_Volume_Tight
Hypothesis: Trade Camarilla pivot breakouts on 4h with volume confirmation and 1d trend filter. Enter long when price breaks above R1 with volume > 2x 24-period average and 1d EMA34 trending up. Enter short when price breaks below S1 with volume > 2x average and 1d EMA34 trending down. Tight conditions target 20-40 trades/year. Works in bull/bear by following 1d trend direction. Uses volume spike to avoid false breakouts.
"""

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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA34 for trend filter
    ema_period = 34
    ema_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period-1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 / (ema_period + 1)) + (ema_1d[i-1] * (ema_period - 1) / (ema_period + 1))
    
    # Align 1d EMA to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Get 4h data for Camarilla pivots
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels from previous 4h bar
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_R1 = np.full_like(close_4h, np.nan)
    camarilla_S1 = np.full_like(close_4h, np.nan)
    
    for i in range(1, len(close_4h)):
        if not (np.isnan(high_4h[i-1]) or np.isnan(low_4h[i-1]) or np.isnan(close_4h[i-1])):
            camarilla_R1[i] = close_4h[i-1] + (high_4h[i-1] - low_4h[i-1]) * 1.1 / 12
            camarilla_S1[i] = close_4h[i-1] - (high_4h[i-1] - low_4h[i-1]) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_S1)
    
    # Volume confirmation: volume > 2x 24-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(35, vol_period)  # EMA needs ~34 periods
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(camarilla_R1_aligned[i]) or 
            np.isnan(camarilla_S1_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Trend filter: 1d EMA34 slope
        if i > 0 and not np.isnan(ema_1d_aligned[i-1]):
            ema_trend_up = ema_1d_aligned[i] > ema_1d_aligned[i-1]
            ema_trend_down = ema_1d_aligned[i] < ema_1d_aligned[i-1]
        else:
            ema_trend_up = False
            ema_trend_down = False
        
        if position == 0:
            # Long: price > R1 + volume + uptrend
            if close[i] > camarilla_R1_aligned[i] and vol_confirm and ema_trend_up:
                signals[i] = 0.25
                position = 1
            # Short: price < S1 + volume + downtrend
            elif close[i] < camarilla_S1_aligned[i] and vol_confirm and ema_trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price < S1 or downtrend
            if close[i] < camarilla_S1_aligned[i] or (i > 0 and not np.isnan(ema_1d_aligned[i-1]) and ema_1d_aligned[i] < ema_1d_aligned[i-1]):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > R1 or uptrend
            if close[i] > camarilla_R1_aligned[i] or (i > 0 and not np.isnan(ema_1d_aligned[i-1]) and ema_1d_aligned[i] > ema_1d_aligned[i-1]):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_Tight"
timeframe = "4h"
leverage = 1.0