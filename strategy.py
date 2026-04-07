#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour price action with weekly pivot points and volume confirmation
# Long when price breaks above weekly R4 pivot with volume > 2x 20-period average
# Short when price breaks below weekly S4 pivot with volume > 2x 20-period average
# Exit when price crosses opposite weekly pivot level (R3/S3) or stoploss at 2.5 * ATR
# Weekly pivot provides institutional reference point for 60-90% of price moves
# Volume filter ensures breakouts have institutional participation
# Target: 80-150 total trades over 4 years (20-38/year)

name = "6h_weekly_pivot_vol_v1"
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
    
    # Weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points: (H+L+C)/3
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Calculate R4 and S4 levels
    weekly_range = weekly_high - weekly_low
    r4 = weekly_pivot + 3 * weekly_range
    s4 = weekly_pivot - 3 * weekly_range
    
    # Calculate R3 and S3 for exit levels
    r3 = weekly_pivot + 2 * weekly_range
    s3 = weekly_pivot - 2 * weekly_range
    
    # Align weekly levels to 6h timeframe (shifted by 1 for completed week only)
    r4_aligned = align_htf_to_ltf(prices, df_weekly, r4)
    s4_aligned = align_htf_to_ltf(prices, df_weekly, s4)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Average volume for volume confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below weekly R3
            elif close[i] < r3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above weekly S3
            elif close[i] > s3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Volume confirmation: current volume > 2 * average volume
            volume_confirm = volume[i] > 2.0 * vol_avg[i]
            
            # Long: price breaks above weekly R4 with volume
            if close[i] > r4_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below weekly S4 with volume
            elif close[i] < s4_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals