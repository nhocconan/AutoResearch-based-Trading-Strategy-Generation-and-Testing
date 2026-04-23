#!/usr/bin/env python3
"""
Hypothesis: 6h strategy combining weekly Camarilla pivot levels with 1d EMA34 trend filter and volume spike confirmation.
Long when price breaks above weekly R3 level and close > 1d EMA34 (uptrend) with volume > 2.0x average.
Short when price breaks below weekly S3 level and close < 1d EMA34 (downtrend) with volume > 2.0x average.
Exit on opposite Camarilla level (R2/S2) break or trend reversal. Uses 6h timeframe targeting 50-150 total trades over 4 years.
Weekly Camarilla provides strong support/resistance from smart money levels, 1d EMA34 filters intermediate trend,
volume spike confirms institutional participation. Designed to capture strong momentum moves while avoiding whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for Camarilla calculation - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels (based on previous week)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low),
    #            S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    weekly_range = high_1w - low_1w
    r3_1w = close_1w + 1.1 * weekly_range
    s3_1w = close_1w - 1.1 * weekly_range
    r2_1w = close_1w + 0.5 * weekly_range  # Exit level for longs
    s2_1w = close_1w - 0.5 * weekly_range  # Exit level for shorts
    
    # Load 1d data for EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 6h timeframe
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or 
            np.isnan(r2_1w_aligned[i]) or np.isnan(s2_1w_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r3_val = r3_1w_aligned[i]
        s3_val = s3_1w_aligned[i]
        r2_val = r2_1w_aligned[i]
        s2_val = s2_1w_aligned[i]
        ema34_val = ema34_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above weekly R3 AND price > 1d EMA34 (uptrend) AND volume spike
            if (price > r3_val and price > ema34_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below weekly S3 AND price < 1d EMA34 (downtrend) AND volume spike
            elif (price < s3_val and price < ema34_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below weekly R2 OR trend reversal
                if (price < r2_val or price < ema34_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above weekly S2 OR trend reversal
                if (price > s2_val or price > ema34_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WeeklyCamarilla_R3_S3_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0