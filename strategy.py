#!/usr/bin/env python3
"""
6h Camarilla Pivot with 1d EMA Trend Filter and Volume Spike
Hypothesis: Camarilla pivot levels from daily timeframe act as strong support/resistance.
Price breaking above R4 or below S4 with trend alignment (1d EMA50) and volume confirmation
captures strong momentum moves. Fading at R3/S3 provides counter-trend opportunities in ranging markets.
Works in both bull and bear by following higher timeframe trend.
Target: 15-30 trades/year per symbol to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_trend_volume_v1"
timeframe = "6h"
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
    
    # Volume Spike Detector (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    # 1d EMA50 Trend Filter
    df_1d = get_htf_data(prices, '1d')
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla formulas: 
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # R2 = C + ((H-L) * 1.1/6)
    # R1 = C + ((H-L) * 1.1/12)
    # PP = (H + L + C) / 3
    # S1 = C - ((H-L) * 1.1/12)
    # S2 = C - ((H-L) * 1.1/6)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    
    # Get previous day's OHLC (need to shift by 1 to avoid look-ahead)
    prev_day_open = df_1d['open'].shift(1).values
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    prev_day_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    H_L = prev_day_high - prev_day_low
    camarilla_r4 = prev_day_close + (H_L * 1.1 / 2)
    camarilla_r3 = prev_day_close + (H_L * 1.1 / 4)
    camarilla_s3 = prev_day_close - (H_L * 1.1 / 4)
    camarilla_s4 = prev_day_close - (H_L * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below 1d EMA50 OR touches R3 (take profit)
            if close[i] < ema_50_aligned[i] or close[i] >= r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above 1d EMA50 OR touches S3 (take profit)
            if close[i] > ema_50_aligned[i] or close[i] <= s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long: price breaks above R4 with trend alignment and volume spike
            if (close[i] > r4_aligned[i-1] and 
                close[i] > ema_50_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Breakout short: price breaks below S4 with trend alignment and volume spike
            elif (close[i] < s4_aligned[i-1] and 
                  close[i] < ema_50_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            # Fade long: price touches S3 with counter-trend (price below EMA) and volume spike
            elif (close[i] <= s3_aligned[i] and 
                  close[i] < ema_50_aligned[i] and 
                  vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Fade short: price touches R3 with counter-trend (price above EMA) and volume spike
            elif (close[i] >= r3_aligned[i] and 
                  close[i] > ema_50_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals