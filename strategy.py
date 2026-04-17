#!/usr/bin/env python3
"""
1D Weekly Range Breakout with Volume Confirmation and Trend Filter
Long: Close breaks above prior weekly high AND volume > 2x daily volume SMA(20) AND price > 1w EMA(50)
Short: Close breaks below prior weekly low AND volume > 2x daily volume SMA(20) AND price < 1w EMA(50)
Exit: Opposite breakout or volume drops below average
Uses weekly range for structure, volume for confirmation, weekly EMA for trend filter
Target: 10-25 trades/year per symbol (40-100 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for range and trend
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly high and low (prior week's values)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_high_shifted = np.roll(weekly_high, 1)
    weekly_low_shifted = np.roll(weekly_low, 1)
    weekly_high_shifted[0] = np.nan
    weekly_low_shifted[0] = np.nan
    
    # Align weekly levels to daily
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high_shifted)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low_shifted)
    
    # Calculate weekly EMA(50) for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily volume SMA(20) for volume filter
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # Need enough data for EMA50 and volume SMA
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma[i]
        weekly_high_val = weekly_high_aligned[i]
        weekly_low_val = weekly_low_aligned[i]
        ema_50_val = ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: Close breaks above weekly high + volume > 2x SMA + price > weekly EMA50
            if close[i] > weekly_high_val and vol > 2.0 * vol_sma_val and price > ema_50_val:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below weekly low + volume > 2x SMA + price < weekly EMA50
            elif close[i] < weekly_low_val and vol > 2.0 * vol_sma_val and price < ema_50_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Close breaks below weekly low OR volume drops below average
            if close[i] < weekly_low_val or vol < 0.5 * vol_sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close breaks above weekly high OR volume drops below average
            if close[i] > weekly_high_val or vol < 0.5 * vol_sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1D_Weekly_Range_Breakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0