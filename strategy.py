#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Camarilla_R3S3_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's high, low, close for Camarilla calculation
    phigh = np.roll(high_1d, 1)
    plow = np.roll(low_1d, 1)
    pclose = np.roll(close_1d, 1)
    phigh[0] = high_1d[0]
    plow[0] = low_1d[0]
    pclose[0] = close_1d[0]
    
    # Camarilla levels: R3, S3
    # R3 = close + 1.1*(high-low)
    # S3 = close - 1.1*(high-low)
    camarilla_r3 = pclose + 1.1 * (phigh - plow)
    camarilla_s3 = pclose - 1.1 * (phigh - plow)
    
    # Align Camarilla levels to daily timeframe (same timeframe, so no shift needed)
    camarilla_r3_aligned = camarilla_r3  # already aligned to daily
    camarilla_s3_aligned = camarilla_s3  # already aligned to daily
    
    # Volume spike: current volume > 2.0x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough history for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Camarilla R3, price above 1w EMA34, volume spike
            long_cond = (close[i] > camarilla_r3_aligned[i] and 
                        close[i] > ema34_1w_aligned[i] and
                        volume_spike[i])
            
            # Short: Price breaks below Camarilla S3, price below 1w EMA34, volume spike
            short_cond = (close[i] < camarilla_s3_aligned[i] and 
                         close[i] < ema34_1w_aligned[i] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price closes below Camarilla S3 OR price crosses below 1w EMA34
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price closes above Camarilla R3 OR price crosses above 1w EMA34
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals