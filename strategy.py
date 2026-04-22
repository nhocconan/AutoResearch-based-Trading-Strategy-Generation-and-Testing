#!/usr/bin/env python3
"""
Hypothesis: 4-hour Camarilla pivot levels with 1-day trend filter and volume spike confirmation.
Long when price breaks above R1 in uptrend with volume spike. Short when price breaks below S1 in downtrend with volume spike.
Exit when price returns to pivot point (P). Uses actual Camarilla calculation from prior day's high-low-close.
Camarilla levels provide institutional support/resistance; volume filter ensures follow-through; trend filter avoids counter-trend trades.
Works in bull markets (breakouts continue) and bear markets (breakdowns continue) by following institutional levels.
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
    
    # Load 1-day data for Camarilla levels and trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), R2 = C + ((H-L)*1.1/6), R1 = C + ((H-L)*1.1/12)
    # S1 = C - ((H-L)*1.1/12), S2 = C - ((H-L)*1.1/6), S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    # P = (H + L + C) / 3
    
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Calculate levels for each day
    camarilla_p = np.full(len(df_1d), np.nan)
    camarilla_r1 = np.full(len(df_1d), np.nan)
    camarilla_s1 = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if i == 0:  # Need previous day
            continue
        high_prev = h_1d[i-1]
        low_prev = l_1d[i-1]
        close_prev = c_1d[i-1]
        
        camarilla_p[i] = (high_prev + low_prev + close_prev) / 3.0
        camarilla_r1[i] = close_prev + ((high_prev - low_prev) * 1.1 / 12)
        camarilla_s1[i] = close_prev - ((high_prev - low_prev) * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_p_aligned = align_htf_to_ltf(prices, df_1d, camarilla_p)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 1-day EMA trend filter
    ema_34_1d = pd.Series(c_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 2x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_p_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1, uptrend (price > EMA34), volume spike
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1, downtrend (price < EMA34), volume spike
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit when price returns to pivot point (P)
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls back to or below pivot
                if close[i] <= camarilla_p_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises back to or above pivot
                if close[i] >= camarilla_p_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0