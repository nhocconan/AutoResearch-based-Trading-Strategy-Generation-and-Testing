#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_Camarilla_R3_S3_Breakout_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for Camarilla pivot levels (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Previous week's Camarilla pivot levels
    prev_close = np.roll(close_1w, 1)
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close[0] = close_1w[0]
    prev_high[0] = high_1w[0]
    prev_low[0] = low_1w[0]
    
    # Weekly Camarilla R3 and S3 levels
    R3_w = prev_close + 1.1 * (prev_high - prev_low) / 4
    S3_w = prev_close - 1.1 * (prev_high - prev_low) / 4
    
    # Align weekly Camarilla levels to daily timeframe
    R3_w_aligned = align_htf_to_ltf(prices, df_1w, R3_w)
    S3_w_aligned = align_htf_to_ltf(prices, df_1w, S3_w)
    
    # Weekly EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume spike: current volume > 2.0x 20-period average (daily)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(R3_w_aligned[i]) or np.isnan(S3_w_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly R3, price above weekly EMA34, volume spike
            long_cond = (close[i] > R3_w_aligned[i] and 
                        close[i] > ema34_1w_aligned[i] and
                        volume_spike[i])
            
            # Short: Price breaks below weekly S3, price below weekly EMA34, volume spike
            short_cond = (close[i] < S3_w_aligned[i] and 
                         close[i] < ema34_1w_aligned[i] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price breaks below weekly S3 OR price crosses below weekly EMA34
            if close[i] < S3_w_aligned[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above weekly R3 OR price crosses above weekly EMA34
            if close[i] > R3_w_aligned[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals