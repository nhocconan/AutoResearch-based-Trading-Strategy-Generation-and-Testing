#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # 1D data for trend filter and Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Camarilla pivot levels
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    R4 = prev_close + ((prev_high - prev_low) * 1.1 / 2)
    R3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    S3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    S4 = prev_close - ((prev_high - prev_low) * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    R3_6h = align_htf_to_ltf(prices, df_1d, R3)
    S3_6h = align_htf_to_ltf(prices, df_1d, S3)
    R4_6h = align_htf_to_ltf(prices, df_1d, R4)
    S4_6h = align_htf_to_ltf(prices, df_1d, S4)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 1.8x 20-period average
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(R3_6h[i]) or np.isnan(S3_6h[i]) or np.isnan(R4_6h[i]) or 
            np.isnan(S4_6h[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above R3 with volume and uptrend
            long_cond = (high[i] > R3_6h[i] and 
                        close[i] > R3_6h[i] and
                        ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and
                        volume_spike[i])
            
            # Short breakdown: price breaks below S3 with volume and downtrend
            short_cond = (low[i] < S3_6h[i] and 
                         close[i] < S3_6h[i] and
                         ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops back below R3 or trend reverses
            if close[i] < R3_6h[i] or ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises back above S3 or trend reverses
            if close[i] > S3_6h[i] or ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals