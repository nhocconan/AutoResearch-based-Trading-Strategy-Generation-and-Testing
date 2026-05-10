#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels (R3/S3) from daily timeframe provide strong support/resistance.
Breakouts above R3 or below S3 with volume confirmation and daily trend filter (EMA34) capture
institutional moves. Works in bull markets (buying strength) and bear markets (selling pressure).
Target: 50-150 total trades over 4 years (12-37/year).
"""

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels for each daily bar
    # R3 = Close + (High - Low) * 1.1/2
    # S3 = Close - (High - Low) * 1.1/2
    cam_r3 = np.full(len(close_1d), np.nan)
    cam_s3 = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if not (np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i])):
            cam_r3[i] = close_1d[i] + (high_1d[i] - low_1d[i]) * 1.1 / 2
            cam_s3[i] = close_1d[i] - (high_1d[i] - low_1d[i]) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (wait for daily close)
    cam_r3_aligned = align_htf_to_ltf(prices, df_1d, cam_r3)
    cam_s3_aligned = align_htf_to_ltf(prices, df_1d, cam_s3)
    
    # Daily EMA34 for trend filter
    if len(close_1d) >= 34:
        ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    else:
        ema34_1d = np.full(len(close_1d), np.nan)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Daily volume SMA20 for volume confirmation
    if len(volume_1d) >= 20:
        vol_sma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    else:
        vol_sma20_1d = np.full(len(volume_1d), np.nan)
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for EMA34
    
    for i in range(start_idx, n):
        if np.isnan(cam_r3_aligned[i]) or np.isnan(cam_s3_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average daily volume (scaled to 12h)
        # Approximate 12h volume from daily: daily volume / 2 (since 24h/12h = 2)
        vol_12h_approx = vol_sma20_1d_aligned[i] / 2.0
        volume_confirm = volume[i] > 1.5 * vol_12h_approx
        
        if position == 0:
            # Long: Break above R3 with uptrend and volume confirmation
            if close[i] > cam_r3_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 with downtrend and volume confirmation
            elif close[i] < cam_s3_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price re-enters Camarilla range (between S3 and R3) or trend reversal
            if close[i] < cam_r3_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price re-enters Camarilla range or trend reversal
            if close[i] > cam_s3_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals