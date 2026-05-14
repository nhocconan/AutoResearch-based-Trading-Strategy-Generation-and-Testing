#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_Breakout_Volume_Trend
Hypothesis: Uses 1-day Camarilla pivot levels for breakout entries with volume confirmation and 4-hour EMA trend filter.
Designed to capture institutional breakouts with institutional volume during trending periods.
Works in bull/bear by following 4h trend direction - avoids counter-trend losses.
Targets 20-40 trades/year per symbol with focus on high-probability setups.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Pivot_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA20 for trend filter
    ema_20_4h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from previous day
    # Camarilla: H4 = C + 1.5*(H-L), L4 = C - 1.5*(H-L), etc.
    # But we use simpler version: R3 = H + 1.1*(H-L), S3 = L - 1.1*(H-L)
    # Actually standard Camarilla: R4 = C + (H-L)*1.1/2, S4 = C - (H-L)*1.1/2
    # We'll use: Resistance = C + (H-L)*1.1/2, Support = C - (H-L)*1.1/2
    # But let's use the common version: R3 = H + 1.1*(H-L), S3 = L - 1.1*(H-L)
    
    # Calculate Camarilla levels from daily data
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Camarilla levels: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align Camarilla levels to 4h timeframe (wait for daily close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_20_4h[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_filter = volume[i] > 1.3 * vol_ma_20[i]
        
        # Trend filter: price above/below 4h EMA20
        uptrend = close[i] > ema_20_4h[i]
        downtrend = close[i] < ema_20_4h[i]
        
        # Breakout conditions using Camarilla levels
        breakout_up = close[i] > camarilla_r3_aligned[i]  # Break above R3
        breakdown_down = close[i] < camarilla_s3_aligned[i]  # Break below S3
        
        # Entry conditions: only trade in direction of 4h trend
        long_entry = breakout_up and volume_filter and uptrend
        short_entry = breakdown_down and volume_filter and downtrend
        
        # Exit conditions: return to opposite Camarilla level or trend reversal
        long_exit = (close[i] < camarilla_s3_aligned[i]) or (not uptrend)  # Break below S3 or trend change
        short_exit = (close[i] > camarilla_r3_aligned[i]) or (not downtrend)  # Break above R3 or trend change
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals