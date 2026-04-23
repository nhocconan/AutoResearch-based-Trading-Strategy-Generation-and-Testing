#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R4/S4 breakout with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above 6h Camarilla R4 level AND price > 1d EMA50 (uptrend) AND volume > 1.8x average.
Short when price breaks below 6h Camarilla S4 level AND price < 1d EMA50 (downtrend) AND volume > 1.8x average.
Exit when price reverts to 6h Camarilla pivot point (PP) or trend reverses (price crosses 1d EMA50).
Uses 6h timeframe to reduce trade frequency vs lower timeframes, targeting 50-150 total trades over 4 years.
1d EMA50 provides trend filter. Volume spike ensures high-conviction breakouts.
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
    
    # Calculate 6h Camarilla levels (R4, S4, PP) - ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 10:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Camarilla levels on 6h (based on previous 6h bar's OHLC)
    prev_high_6h = np.roll(high_6h, 1)
    prev_low_6h = np.roll(low_6h, 1)
    prev_close_6h = np.roll(close_6h, 1)
    # Set first value to NaN (no previous bar)
    prev_high_6h[0] = np.nan
    prev_low_6h[0] = np.nan
    prev_close_6h[0] = np.nan
    
    camarilla_pp = (prev_high_6h + prev_low_6h + prev_close_6h) / 3.0
    camarilla_r4 = prev_close_6h + (prev_high_6h - prev_low_6h) * 1.1 / 2.0 * 2.0  # R4 = R3 * 2
    camarilla_s4 = prev_close_6h - (prev_high_6h - prev_low_6h) * 1.1 / 2.0 * 2.0  # S4 = S3 * 2
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_6h, camarilla_pp)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_6h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_6h, camarilla_s4)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma_primary = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_primary[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pp_val = camarilla_pp_aligned[i]
        r4_val = camarilla_r4_aligned[i]
        s4_val = camarilla_s4_aligned[i]
        ema50_val = ema50_1d_aligned[i]
        vol_ma_val = vol_ma_primary[i]
        
        # Get current price and volume
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above 6h Camarilla R4 AND price > 1d EMA50 (uptrend) AND volume spike
            if (price > r4_val and price > ema50_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 6h Camarilla S4 AND price < 1d EMA50 (downtrend) AND volume spike
            elif (price < s4_val and price < ema50_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to Camarilla PP OR price breaks below 1d EMA50 (trend reversal)
                if price <= pp_val or price < ema50_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to Camarilla PP OR price breaks above 1d EMA50 (trend reversal)
                if price >= pp_val or price > ema50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R4_S4_Breakout_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0