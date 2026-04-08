#!/usr/bin/env python3
"""
12h Camarilla Pivot with 1d Trend Filter and Volume Confirmation
Hypothesis: Camarilla pivot reversals filtered by 1d EMA trend and volume spikes yield high-probability mean-reversion trades.
Works in both bull and bear markets by fading extremes during ranging conditions while capturing trend continuations.
Targets 12-37 trades/year with low turnover to minimize fee drag.
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
    
    # Calculate Camarilla pivot levels from previous 12h bar
    # Typical price = (H + L + C) / 3
    typical_price = (high + low + close) / 3.0
    # Range = H - L
    daily_range = high - low
    # Camarilla levels for intraday trading
    # Resistance levels: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6, R1 = C + (H-L)*1.1/12
    # Support levels: S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    r1 = close + (daily_range * 1.1 / 12)
    r2 = close + (daily_range * 1.1 / 6)
    r3 = close + (daily_range * 1.1 / 4)
    r4 = close + (daily_range * 1.1 / 2)
    s1 = close - (daily_range * 1.1 / 12)
    s2 = close - (daily_range * 1.1 / 6)
    s3 = close - (daily_range * 1.1 / 4)
    s4 = close - (daily_range * 1.1 / 2)
    
    # Shift levels to avoid look-ahead (use previous bar's levels)
    r1 = np.roll(r1, 1)
    r2 = np.roll(r2, 1)
    r3 = np.roll(r3, 1)
    r4 = np.roll(r4, 1)
    s1 = np.roll(s1, 1)
    s2 = np.roll(s2, 1)
    s3 = np.roll(s3, 1)
    s4 = np.roll(s4, 1)
    # Set first value to NaN
    r1[0] = r2[0] = r3[0] = r4[0] = s1[0] = s2[0] = s3[0] = s4[0] = np.nan
    
    # Volume filter: current volume > 1.5x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(r1[i]) or np.isnan(r2[i]) or np.isnan(r3[i]) or np.isnan(r4[i]) or
            np.isnan(s1[i]) or np.isnan(s2[i]) or np.isnan(s3[i]) or np.isnan(s4[i]) or
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches S3 (strong support) or trend reverses
            if (close[i] <= s3[i] or 
                close[i] < ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches R3 (strong resistance) or trend reverses
            if (close[i] >= r3[i] or 
                close[i] > ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter: price vs 1d EMA50
            uptrend = close[i] > ema_50_1d_aligned[i]
            downtrend = close[i] < ema_50_1d_aligned[i]
            
            # Long: price touches S1 with uptrend and volume spike (bounce from support in uptrend)
            if (close[i] <= s1[i] and 
                uptrend and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: price touches R1 with downtrend and volume spike (rejection from resistance in downtrend)
            elif (close[i] >= r1[i] and 
                  downtrend and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals