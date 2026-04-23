#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume confirmation.
Long when price breaks above R3 AND price > 4h EMA34 AND volume > 1.3x average.
Short when price breaks below S3 AND price < 4h EMA34 AND volume > 1.3x average.
Exit on opposite Camarilla level touch or volume drop below average.
Uses 4h for signal direction (trend filter + Camarilla calculation), 1h only for entry timing precision.
Targets 15-35 trades/year to minimize fee drag while capturing intraday momentum in both bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for HTF calculations - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA34 for trend filter
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Calculate 4h Camarilla levels (based on previous 4h bar)
    close_4h_prev = np.roll(close_4h, 1)
    high_4h_prev = np.roll(high_4h, 1)
    low_4h_prev = np.roll(low_4h, 1)
    close_4h_prev[0] = np.nan
    high_4h_prev[0] = np.nan
    low_4h_prev[0] = np.nan
    
    # Camarilla R3, S3 levels: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    camarilla_range = 1.1 * (high_4h_prev - low_4h_prev) / 2
    r3_4h = close_4h_prev + camarilla_range
    s3_4h = close_4h_prev - camarilla_range
    
    # Align Camarilla levels to 1h timeframe
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    
    # Volume average (24-period = 6h) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema34_4h_aligned[i]) or np.isnan(r3_4h_aligned[i]) or 
            np.isnan(s3_4h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_val = ema34_4h_aligned[i]
        r3_val = r3_4h_aligned[i]
        s3_val = s3_4h_aligned[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above R3 AND price > 4h EMA34 AND volume spike
            if (price > r3_val and price > ema34_val and vol_current > 1.3 * vol_ma_val):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 AND price < 4h EMA34 AND volume spike
            elif (price < s3_val and price < ema34_val and vol_current > 1.3 * vol_ma_val):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price touches S3 level OR volume drops below average
                if (price <= s3_val or vol_current < vol_ma_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price touches R3 level OR volume drops below average
                if (price >= r3_val or vol_current < vol_ma_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R3S3_4hEMA34_Volume"
timeframe = "1h"
leverage = 1.0