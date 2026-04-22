#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 12h trend filter and volume spike
    # Uses actual pivot levels (not OHLC-based) from prior day for structure
    # 12h EMA50 filters trend direction, volume surge confirms breakout strength
    # Works in bull/bear: breakouts from key levels with momentum capture moves
    
    # Load 12h data once
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA50 trend filter
    ema_12h_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_50_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_50)
    
    # Load daily data for Camarilla pivots (based on prior day OHLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day (based on prior day)
    # R3, R2, R1, S1, S2, S3
    # R4 = close + 1.1*(high-low)*1.5, R3 = close + 1.1*(high-low)*1.25, etc.
    # But standard: R3 = close + 1.1*(high-low)*1.1, R2 = close + 1.1*(high-low)*0.55, R1 = close + 1.1*(high-low)*0.275
    # S1 = close - 1.1*(high-low)*0.275, S2 = close - 1.1*(high-low)*0.55, S3 = close - 1.1*(high-low)*1.1
    
    # We'll use R1/S1 as primary breakout levels
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) * 0.275
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) * 0.275
    
    # Align to 4h - each daily level applies to all 4h bars of that day
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Price and volume data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_12h_50_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R1 with volume spike and 12h uptrend
            if close[i] > camarilla_r1_aligned[i] and vol_spike[i] and close[i] > ema_12h_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with volume spike and 12h downtrend
            elif close[i] < camarilla_s1_aligned[i] and vol_spike[i] and close[i] < ema_12h_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to pivot point (close of prior day) or opposite level touch
            # Pivot point = (high + low + close)/3
            pivot_point = (high_1d + low_1d + close_1d) / 3.0
            pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
            
            if position == 1:
                if close[i] < pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0