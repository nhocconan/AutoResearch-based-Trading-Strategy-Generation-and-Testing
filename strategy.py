#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1-week Donchian breakout (break above/below weekly high/low),
# volume confirmation (current volume > 2x 20-period average), and trend filter (12h EMA50 > EMA200 for long, < for short).
# Uses weekly structure to capture longer-term moves, reducing whipsaw in ranging markets.
# Designed to work in both bull and bear markets by using breakouts with volume confirmation and trend alignment.
# Target: 12-37 trades/year (50-150 over 4 years) to avoid fee drag.

name = "12h_1w_Donchian20_WeeklyTrend_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === Weekly Donchian Channels (20-period) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate rolling high/low for previous 20 weekly bars (excluding current)
    high_roll = pd.Series(high_1w).rolling(window=20, min_periods=20).max().shift(1).values
    low_roll = pd.Series(low_1w).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align to 12h timeframe
    high_roll_aligned = align_htf_to_ltf(prices, df_1w, high_roll)
    low_roll_aligned = align_htf_to_ltf(prices, df_1w, low_roll)
    
    # === Volume Filter ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === Trend Filter: 12h EMA50 > EMA200 for long, < for short ===
    close_series = pd.Series(prices['close'].values)
    ema50 = close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200 = close_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = prices['close'].iloc[i]
        vol_ratio_val = vol_ratio[i]
        high_val = high_roll_aligned[i]
        low_val = low_roll_aligned[i]
        ema50_val = ema50[i]
        ema200_val = ema200[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or np.isnan(high_val) or 
            np.isnan(low_val) or np.isnan(ema50_val) or np.isnan(ema200_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above weekly high with volume confirmation and uptrend
            if close_val > high_val and vol_ratio_val > 2.0 and ema50_val > ema200_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below weekly low with volume confirmation and downtrend
            elif close_val < low_val and vol_ratio_val > 2.0 and ema50_val < ema200_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below weekly low OR trend breaks down
            if close_val < low_val or ema50_val < ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above weekly high OR trend breaks up
            if close_val > high_val or ema50_val > ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals