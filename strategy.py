#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation.
Long when price breaks above R3 and close > 4h EMA20 (uptrend) with volume > 1.5x average.
Short when price breaks below S3 and close < 4h EMA20 (downtrend) with volume > 1.5x average.
Uses 1h timeframe targeting 60-150 total trades over 4 years. Camarilla levels from 4h provide
intraday support/resistance structure. Volume confirmation ensures breakout conviction.
Trend filter prevents counter-trend trades. Session filter (08-20 UTC) reduces noise.
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
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    # Load 4h data for Camarilla pivot calculation - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla pivot levels from previous 4h bar
    # PP = (H + L + C) / 3
    # R3 = PP + (H - L) * 1.1 / 4
    # S3 = PP - (H - L) * 1.1 / 4
    pp = (high_4h + low_4h + close_4h) / 3.0
    r3 = pp + (high_4h - low_4h) * 1.1 / 4.0
    s3 = pp - (high_4h - low_4h) * 1.1 / 4.0
    
    # Load 4h data for EMA20 trend filter - ONCE before loop
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align HTF indicators to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3)
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Volume average (20-period) on primary timeframe
    vol_ma_primary = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema20_4h_aligned[i]) or 
            np.isnan(vol_ma_primary[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter
        if hours[i] < 8 or hours[i] > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema20_val = ema20_4h_aligned[i]
        vol_ma_val = vol_ma_primary[i]
        
        # Get current price and volume
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND price > 4h EMA20 (uptrend) AND volume confirmation
            if (price > r3_val and price > ema20_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla S3 AND price < 4h EMA20 (downtrend) AND volume confirmation
            elif (price < s3_val and price < ema20_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Camarilla S3 OR price breaks below 4h EMA20 (trend reversal)
                if price < s3_val or price < ema20_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above Camarilla R3 OR price breaks above 4h EMA20 (trend reversal)
                if price > r3_val or price > ema20_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R3_S3_4hEMA20_Volume_Session"
timeframe = "1h"
leverage = 1.0