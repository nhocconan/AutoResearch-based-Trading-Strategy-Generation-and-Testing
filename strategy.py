#!/usr/bin/env python3
"""
6h Camarilla Pivot with 1d EMA Trend and Volume Spike
Hypothesis: Camarilla pivot levels from 1d provide key support/resistance.
Fading at R3/S3 levels in ranging markets, breakout continuation at R4/S4.
Using 1d EMA50 for trend filter and volume spikes for confirmation.
Designed to work in both bull and bear regimes by adapting to price action.
Target: 15-35 trades/year per symbol to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_trend_volume_v2"
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
    vol_spike = volume > (vol_ma * 2.0)  # Higher threshold for fewer trades
    
    # 1d OHLC for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    o_1d = df_1d['open'].values
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    range_1d = h_1d - l_1d
    r4 = c_1d + (range_1d * 1.1 / 2)
    r3 = c_1d + (range_1d * 1.1 / 4)
    s3 = c_1d - (range_1d * 1.1 / 4)
    s4 = c_1d - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 6t timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1d EMA50 Trend Filter
    ema_50 = pd.Series(c_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below 1d EMA50
            if close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above 1d EMA50
            if close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: bounce from S3 with volume spike in uptrend
            if (close[i] > s3_aligned[i] and 
                close[i] < s3_aligned[i] * 1.005 and  # Within 0.5% of S3
                close[i] > ema_50_aligned[i] and
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: rejection at R3 with volume spike in downtrend
            elif (close[i] < r3_aligned[i] and 
                  close[i] > r3_aligned[i] * 0.995 and  # Within 0.5% of R3
                  close[i] < ema_50_aligned[i] and
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            # Long breakout: close above R4 with volume spike
            elif (close[i] > r4_aligned[i] and 
                  vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown: close below S4 with volume spike
            elif (close[i] < s4_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals