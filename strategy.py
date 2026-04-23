#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above 12h Camarilla R3 level AND price > 1d EMA50 (uptrend) AND volume > 1.8x average.
Short when price breaks below 12h Camarilla S3 level AND price < 1d EMA50 (downtrend) AND volume > 1.8x average.
Exit when price reverts to 12h Camarilla H4/L4 level or trend reverses (price crosses 1d EMA50).
Uses 12h timeframe with tight entry conditions to avoid fee drag. Camarilla levels provide precise intraday support/resistance.
1d EMA50 provides stable trend filter. Volume confirmation ensures high-conviction breakouts.
Target: 50-150 trades over 4 years (12-37/year) to stay within proven working range for 12h.
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
    
    # Calculate 12h Camarilla levels - using primary timeframe data
    # We need to calculate on 12h data but we only have 15m/1h etc. - so we'll use rolling on current timeframe
    # However, for true 12h Camarilla we need to use 12h data via mtf_data
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for each 12h bar
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    # Where C = close, H = high, L = low of previous day
    # For intraday, we use the previous 12h bar's OHLC
    prev_close_12h = np.roll(close_12h, 1)
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    
    # Set first value to NaN since no previous bar
    prev_close_12h[0] = np.nan
    prev_high_12h[0] = np.nan
    prev_low_12h[0] = np.nan
    
    # Calculate Camarilla levels
    R3_12h = prev_close_12h + ((prev_high_12h - prev_low_12h) * 1.1 / 4)
    S3_12h = prev_close_12h - ((prev_high_12h - prev_low_12h) * 1.1 / 4)
    H4_12h = prev_close_12h + ((prev_high_12h - prev_low_12h) * 1.1 / 2)
    L4_12h = prev_close_12h - ((prev_high_12h - prev_low_12h) * 1.1 / 2)
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to primary timeframe
    R3_12h_aligned = align_htf_to_ltf(prices, df_12h, R3_12h)
    S3_12h_aligned = align_htf_to_ltf(prices, df_12h, S3_12h)
    H4_12h_aligned = align_htf_to_ltf(prices, df_12h, H4_12h)
    L4_12h_aligned = align_htf_to_ltf(prices, df_12h, L4_12h)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma_primary = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(R3_12h_aligned[i]) or np.isnan(S3_12h_aligned[i]) or np.isnan(H4_12h_aligned[i]) or 
            np.isnan(L4_12h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_primary[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        R3_val = R3_12h_aligned[i]
        S3_val = S3_12h_aligned[i]
        H4_val = H4_12h_aligned[i]
        L4_val = L4_12h_aligned[i]
        ema50_val = ema50_1d_aligned[i]
        vol_ma_val = vol_ma_primary[i]
        vol_current = volume[i]
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above 12h Camarilla R3 level AND price > 1d EMA50 (uptrend) AND volume confirmation
            if (price > R3_val and price > ema50_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Camarilla S3 level AND price < 1d EMA50 (downtrend) AND volume confirmation
            elif (price < S3_val and price < ema50_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to 12h Camarilla H4 level OR price breaks below 1d EMA50 (trend reversal)
                if price >= H4_val or price < ema50_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to 12h Camarilla L4 level OR price breaks above 1d EMA50 (trend reversal)
                if price <= L4_val or price > ema50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_R3_S3_Breakout_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0