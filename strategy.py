#!/usr/bin/env python3
"""
1d_camarilla_pivot_1w_trend_volume_v1
Hypothesis: On 1d timeframe, use Camarilla pivot levels from weekly timeframe (R3/S3) for entry/exit, with trend filter from weekly EMA200 and volume confirmation. Enter long when price crosses above S3 with weekly uptrend and volume > 1.5x average. Enter short when price crosses below R3 with weekly downtrend and volume > 1.5x average. Exit when price crosses opposite pivot level or trend reverses. Targets 10-25 trades/year to minimize fee drift.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_camarilla_pivot_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for Camarilla pivots and trend filter (calculate once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from weekly OHLC
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # We'll use R3 and S3 for entries
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate pivot levels
    cam_r3 = weekly_close + ((weekly_high - weekly_low) * 1.1 / 4)
    cam_s3 = weekly_close - ((weekly_high - weekly_low) * 1.1 / 4)
    
    # Calculate weekly EMA200 for trend filter
    weekly_close_s = pd.Series(weekly_close)
    ema200_1w = weekly_close_s.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Align to daily timeframe (shifted by 1 week to avoid look-ahead)
    cam_r3_aligned = align_htf_to_ltf(prices, df_1w, cam_r3)
    cam_s3_aligned = align_htf_to_ltf(prices, df_1w, cam_s3)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if required data not available
        if (np.isnan(cam_r3_aligned[i]) or np.isnan(cam_s3_aligned[i]) or 
            np.isnan(ema200_1w_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter from weekly: up if close > EMA200, down if close < EMA200
        trend_up = close[i] > ema200_1w_aligned[i]
        trend_down = close[i] < ema200_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when price crosses below S3 (mean reversion)
            if close[i] < cam_s3_aligned[i]:
                exit_long = True
            # Exit on trend reversal (price below EMA200)
            elif close[i] < ema200_1w_aligned[i]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit when price crosses above R3 (mean reversion)
            if close[i] > cam_r3_aligned[i]:
                exit_short = True
            # Exit on trend reversal (price above EMA200)
            elif close[i] > ema200_1w_aligned[i]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price crosses above S3, weekly uptrend, volume confirmation
            long_entry = (close[i] > cam_s3_aligned[i] and 
                         close[i-1] <= cam_s3_aligned[i-1] and  # crossed above
                         trend_up and vol_confirm)
            
            # Short entry: price crosses below R3, weekly downtrend, volume confirmation
            short_entry = (close[i] < cam_r3_aligned[i] and 
                          close[i-1] >= cam_r3_aligned[i-1] and  # crossed below
                          trend_down and vol_confirm)
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals