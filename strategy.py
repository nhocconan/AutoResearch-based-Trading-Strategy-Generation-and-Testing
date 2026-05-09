#!/usr/bin/env python3
# Hypothesis: 6h volume-weighted RSI divergence with weekly trend filter
# Long when RSI < 30 on declining volume (bullish divergence) and weekly close > weekly open
# Short when RSI > 70 on declining volume (bearish divergence) and weekly close < weekly open
# Exit when RSI returns to 50 or opposite divergence occurs
# Uses volume confirmation to filter false signals and weekly trend for direction bias
# Designed to work in both bull (buy dips) and bear (sell rallies) markets
# Target: 60-120 total trades over 4 years (15-30/year) with size 0.25

name = "6h_VolumeWeightedRSI_Divergence_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prrices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly trend filter (weekly close > weekly open = bullish)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    weekly_bullish = df_1w['close'] > df_1w['open']
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.values.astype(float))
    
    # Calculate RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate volume-weighted RSI (emphasize RSI moves on high volume)
    vol_ratio = volume / (pd.Series(volume).rolling(window=20, min_periods=20).mean().values + 1e-10)
    vol_weighted_rsi = rsi * vol_ratio
    vol_weighted_rsi_smooth = pd.Series(vol_weighted_rsi).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Volume declining condition (current volume < previous volume)
    vol_declining = volume < np.roll(volume, 1)
    vol_declining[0] = False  # First value has no previous
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for RSI calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_bullish_aligned[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(vol_weighted_rsi_smooth[i]) or np.isnan(vol_declining[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: RSI < 30, volume declining (bullish divergence), weekly bullish
            if (rsi_values[i] < 30 and vol_declining[i] and weekly_bullish_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Enter short: RSI > 70, volume declining (bearish divergence), weekly bearish
            elif (rsi_values[i] > 70 and vol_declining[i] and weekly_bullish_aligned[i] <= 0.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to 50 or bearish divergence appears
            if (rsi_values[i] >= 50) or (rsi_values[i] > 70 and vol_declining[i] and weekly_bullish_aligned[i] <= 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to 50 or bullish divergence appears
            if (rsi_values[i] <= 50) or (rsi_values[i] < 30 and vol_declining[i] and weekly_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals