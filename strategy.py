#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels provide institutional reference points. 
Breakouts above R3 or below S3 with 1d trend alignment and volume surge capture 
strong momentum moves. Works in bull markets via upside breakouts and in bear 
markets via downside breakdowns. Uses 12h timeframe to limit trades and avoid 
fee drag, with 1d trend and volume filters for confirmation.
"""

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
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
    
    # Calculate 12h Camarilla levels (using previous bar's range)
    camarilla_r3 = np.zeros(n)
    camarilla_s3 = np.zeros(n)
    camarilla_r4 = np.zeros(n)
    camarilla_s4 = np.zeros(n)
    
    for i in range(1, n):
        prev_close = close[i-1]
        prev_range = high[i-1] - low[i-1]
        
        camarilla_r3[i] = prev_close + 1.1 * prev_range / 6
        camarilla_s3[i] = prev_close - 1.1 * prev_range / 6
        camarilla_r4[i] = prev_close + 1.1 * prev_range / 2
        camarilla_s4[i] = prev_close - 1.1 * prev_range / 2
    
    # Get 1d data for trend and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34 = np.zeros_like(close_1d)
    ema_34[:] = np.nan
    if len(close_1d) >= 34:
        ema_34[33] = np.mean(close_1d[:34])
        for i in range(34, len(close_1d)):
            ema_34[i] = (close_1d[i] * 2/35) + (ema_34[i-1] * 33/35)
    
    # Calculate 1d volume average (20-period) for volume spike filter
    vol_1d = df_1d['volume'].values
    vol_avg_20 = np.zeros_like(vol_1d)
    vol_avg_20[:] = np.nan
    if len(vol_1d) >= 20:
        for i in range(19, len(vol_1d)):
            vol_avg_20[i] = np.mean(vol_1d[i-19:i+1])
    
    # Align 1d indicators to 12h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_avg_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike: current 12h volume > 1.5x 1d average volume
        vol_spike = volume[i] > 1.5 * vol_avg_20_aligned[i]
        
        if position == 0:
            # LONG: Close above R3 + 1d uptrend + volume spike
            if (close[i] > camarilla_r3[i] and 
                close[i] > ema_34_aligned[i] and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # SHORT: Close below S3 + 1d downtrend + volume spike
            elif (close[i] < camarilla_s3[i] and 
                  close[i] < ema_34_aligned[i] and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below R3 or loss of trend/volume
            if (close[i] < camarilla_r3[i] or 
                close[i] < ema_34_aligned[i] or 
                not vol_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above S3 or loss of trend/volume
            if (close[i] > camarilla_s3[i] or 
                close[i] > ema_34_aligned[i] or 
                not vol_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals