#!/usr/bin/env python3
"""
12h_1d_Camarilla_Pivot_Breakout_Volume_Trend_v1
Hypothesis: Uses 1-day Camarilla pivot levels for breakout entries on 12h timeframe with volume confirmation and daily ATR trend filter.
Targets 15-30 trades/year per symbol with focus on high-probability setups in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_Pivot_Breakout_Volume_Trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla pivots and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 14-day ATR for trend filter (using daily data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Pad with NaN for first element
    tr = np.concatenate([[np.nan], tr])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12-period SMA of ATR for trend direction
    atr_ma_12 = pd.Series(atr_14).rolling(window=12, min_periods=12).mean().values
    
    # Volume filter: 20-period average on 12h timeframe
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from daily data
    # Camarilla levels: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align Camarilla levels to 12h timeframe (wait for daily close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_ma_12[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average (stricter)
        volume_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: ATR rising = trending market, ATR falling = ranging market
        # Using ATR MA slope: rising when current > previous
        trending_up = atr_ma_12[i] > atr_ma_12[i-1] if i > 0 else False
        trending_down = atr_ma_12[i] < atr_ma_12[i-1] if i > 0 else False
        
        # Breakout conditions using Camarilla levels
        breakout_up = close[i] > camarilla_r3_aligned[i]  # Break above R3
        breakdown_down = close[i] < camarilla_s3_aligned[i]  # Break below S3
        
        # Entry conditions: only trade in trending markets
        long_entry = breakout_up and volume_filter and trending_up
        short_entry = breakdown_down and volume_filter and trending_down
        
        # Exit conditions: return to opposite Camarilla level or trend change
        long_exit = (close[i] < camarilla_s3_aligned[i]) or (not trending_up)
        short_exit = (close[i] > camarilla_r3_aligned[i]) or (not trending_down)
        
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