#!/usr/bin/env python3
"""
4h_camarilla_pivot_12h_trend_volume_v1
Hypothesis: Camarilla pivot levels from 12h act as institutional support/resistance.
Price retesting these levels with volume and 12h trend alignment offers high-probability entries.
In bull markets, buy near S3/S4 with trend up; in bear markets, sell near R3/R4 with trend down.
Targets 20-50 trades/year by requiring confluence of pivot retest, volume, and trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_12h_trend_volume_v1"
timeframe = "4h"
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
    
    # 12h OHLC for Camarilla pivots
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 12h bar
    # H, L, C from previous completed 12h bar
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    # Camarilla multipliers
    # Resistance levels: R3 = C + (H-L)*1.1/2, R4 = C + (H-L)*1.1
    # Support levels: S3 = C - (H-L)*1.1/2, S4 = C - (H-L)*1.1
    r3_12h = c_12h + (h_12h - l_12h) * 1.1 / 2
    r4_12h = c_12h + (h_12h - l_12h) * 1.1
    s3_12h = c_12h - (h_12h - l_12h) * 1.1 / 2
    s4_12h = c_12h - (h_12h - l_12h) * 1.1
    
    # Align to 4h (previous 12h bar's levels act as support/resistance)
    r3_4h = align_htf_to_ltf(prices, df_12h, r3_12h)
    r4_4h = align_htf_to_ltf(prices, df_12h, r4_12h)
    s3_4h = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_4h = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_4h = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 20-period volume average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema50_4h[i]) or 
            np.isnan(r3_4h[i]) or 
            np.isnan(r4_4h[i]) or 
            np.isnan(s3_4h[i]) or 
            np.isnan(s4_4h[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below S3 OR trend turns down
            if close[i] < s3_4h[i] or close[i] < ema50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above R3 OR trend turns up
            if close[i] > r3_4h[i] or close[i] > ema50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price near S3/S4 (within 0.5%) + volume + uptrend
            near_s3 = abs(close[i] - s3_4h[i]) / s3_4h[i] < 0.005
            near_s4 = abs(close[i] - s4_4h[i]) / s4_4h[i] < 0.005
            if ((near_s3 or near_s4) and 
                vol_confirm and 
                close[i] > ema50_4h[i]):
                position = 1
                signals[i] = 0.25
            # Short: price near R3/R4 (within 0.5%) + volume + downtrend
            elif ((abs(close[i] - r3_4h[i]) / r3_4h[i] < 0.005 or 
                   abs(close[i] - r4_4h[i]) / r4_4h[i] < 0.005) and 
                  vol_confirm and 
                  close[i] < ema50_4h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals