#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above 4h Camarilla R3 level AND price > 1d EMA34 (uptrend) AND volume > 2.0x average.
Short when price breaks below 4h Camarilla S3 level AND price < 1d EMA34 (downtrend) AND volume > 2.0x average.
Exit when price reverts to 4h Camarilla Pivot level or trend reverses (price crosses 1d EMA34).
Uses 4h timeframe with tight entry conditions to avoid fee drag. Camarilla levels provide precise intraday support/resistance.
1d EMA34 provides stable trend filter. Volume confirmation ensures high-conviction breakouts.
Target: 75-150 trades over 4 years (19-37/year) to stay within proven working range.
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
    
    # Calculate 4h Camarilla levels (based on previous 4h bar's OHLC)
    # We need to calculate on 4h data but we only have primary timeframe data - so we'll use 4h data via mtf_data
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    open_4h = df_4h['open'].values
    
    # Camarilla levels calculation for 4h timeframe
    # Based on previous bar's range: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # But standard Camarilla uses: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    # Pivot = (high + low + close)/3
    range_4h = high_4h - low_4h
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    r3_4h = close_4h + 1.1 * range_4h
    s3_4h = close_4h - 1.1 * range_4h
    
    # Load 1d data for EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to primary timeframe (4h)
    pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma_primary = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(pivot_4h_aligned[i]) or np.isnan(r3_4h_aligned[i]) or np.isnan(s3_4h_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_primary[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pivot_val = pivot_4h_aligned[i]
        r3_val = r3_4h_aligned[i]
        s3_val = s3_4h_aligned[i]
        ema34_val = ema34_1d_aligned[i]
        vol_ma_val = vol_ma_primary[i]
        
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above 4h R3 level AND price > 1d EMA34 (uptrend) AND volume confirmation
            if (price > r3_val and price > ema34_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 4h S3 level AND price < 1d EMA34 (downtrend) AND volume confirmation
            elif (price < s3_val and price < ema34_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to pivot level OR price breaks below 1d EMA34 (trend reversal)
                if price <= pivot_val or price < ema34_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to pivot level OR price breaks above 1d EMA34 (trend reversal)
                if price >= pivot_val or price > ema34_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R3_S3_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0