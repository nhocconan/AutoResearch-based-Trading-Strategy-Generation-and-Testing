#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike.
Long when price breaks above Camarilla R3 AND close > 12h EMA50 AND volume > 2.0x 20-period average.
Short when price breaks below Camarilla S3 AND close < 12h EMA50 AND volume > 2.0x 20-period average.
Exit when price crosses Camarilla H3/L3 levels (mean of R3/S3 and H3/L3).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 20-50 trades/year per symbol.
Camarilla levels provide intraday support/resistance with proven edge on ETH/ETH pairs.
12h EMA50 offers smoother trend filter than 1d EMA34 for 4h timeframe alignment.
Volume confirmation at 2.0x ensures only institutional-grade breakouts are taken.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for Camarilla levels - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels (based on previous 4h bar) on 4h timeframe
    # Camarilla formulas: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    # R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    # Actually: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    #          S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    #          H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    #          H4 = close + 1.5*(high-low)/2, L4 = close - 1.5*(high-low)/2
    # We'll use R3, S3, H3, L3 for breakout and exit
    prev_close_4h = np.concatenate([[np.nan], close_4h[:-1]])
    prev_high_4h = np.concatenate([[np.nan], high_4h[:-1]])
    prev_low_4h = np.concatenate([[np.nan], low_4h[:-1]])
    
    cam_r3 = prev_close_4h + 1.1 * (prev_high_4h - prev_low_4h)
    cam_s3 = prev_close_4h - 1.1 * (prev_high_4h - prev_low_4h)
    cam_h3 = prev_close_4h + 1.1 * (prev_high_4h - prev_low_4h) / 2
    cam_l3 = prev_close_4h - 1.1 * (prev_high_4h - prev_low_4h) / 2
    
    # Align Camarilla levels to 15m timeframe
    cam_r3_aligned = align_htf_to_ltf(prices, df_4h, cam_r3)
    cam_s3_aligned = align_htf_to_ltf(prices, df_4h, cam_s3)
    cam_h3_aligned = align_htf_to_ltf(prices, df_4h, cam_h3)
    cam_l3_aligned = align_htf_to_ltf(prices, df_4h, cam_l3)
    
    # Load 12h data for EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(cam_r3_aligned[i]) or np.isnan(cam_s3_aligned[i]) or 
            np.isnan(cam_h3_aligned[i]) or np.isnan(cam_l3_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND close > 12h EMA50 AND volume spike
            if (price > cam_r3_aligned[i] and 
                close[i] > ema50_12h_aligned[i] and 
                volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 AND close < 12h EMA50 AND volume spike
            elif (price < cam_s3_aligned[i] and 
                  close[i] < ema50_12h_aligned[i] and 
                  volume[i] > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: price crosses Camarilla H3/L3 levels
            if position == 1 and price < cam_h3_aligned[i]:
                exit_signal = True
            elif position == -1 and price > cam_l3_aligned[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R3S3_12hEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0