#!/usr/bin/env python3
"""
12h Camarilla Pivot with 1d Trend and Volume Confirmation
Hypothesis: Price reacting off Camarilla pivot levels (H3/L3) on 12h timeframe,
filtered by 1d EMA trend direction and volume spikes, captures mean-reversion bounces
in ranging markets and breakouts in trending markets. Targets 15-35 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_trend_volume_v1"
timeframe = "12h"
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
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Previous day's OHLC for Camarilla calculation (using 1d data)
    # We need yesterday's high, low, close to calculate today's pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # H3 = close + 1.1*(high - low)/4
    # L3 = close - 1.1*(high - low)/4
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 4
    
    # Align Camarilla levels to 12h timeframe (shifted by 1 day for look-ahead avoidance)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume filter: current volume > 1.5x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below L3 or trend reverses
            if close[i] < camarilla_l3_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above H3 or trend reverses
            if close[i] > camarilla_h3_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter: price vs 1d EMA50
            uptrend = close[i] > ema_50_1d_aligned[i]
            downtrend = close[i] < ema_50_1d_aligned[i]
            
            # Long: price touches/bounces off L3 + uptrend + volume spike
            if (close[i] <= camarilla_l3_aligned[i] * 1.005 and  # Within 0.5% of L3
                uptrend and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: price touches/rejects at H3 + downtrend + volume spike
            elif (close[i] >= camarilla_h3_aligned[i] * 0.995 and  # Within 0.5% of H3
                  downtrend and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals