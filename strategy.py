#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
Long when price breaks above Camarilla R3 AND price > 1d EMA34 AND volume > 2.0x 20-period average.
Short when price breaks below Camarilla S3 AND price < 1d EMA34 AND volume > 2.0x 20-period average.
Exit when price reverts to Camarilla pivot point (PP) OR ATR trailing stop (2.5*ATR from extreme).
Uses 1d HTF for trend alignment. Discrete sizing 0.25 to minimize fee churn.
Target: 75-200 total trades over 4 years (19-50/year).
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough for EMA
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Previous day's OHLC for Camarilla levels (use previous completed day)
    # Roll by 1 to use previous bar's values to avoid look-ahead
    high_roll = np.roll(high, 1)
    low_roll = np.roll(low, 1)
    close_roll = np.roll(close, 1)
    high_roll[0] = np.nan
    low_roll[0] = np.nan
    close_roll[0] = np.nan
    
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low), etc.
    # PP = (high + low + close)/3
    # S3 = close - 1.125*(high-low), S4 = close - 1.5*(high-low)
    hl_range = high_roll - low_roll
    camarilla_pp = (high_roll + low_roll + close_roll) / 3.0
    camarilla_r3 = close_roll + 1.125 * hl_range
    camarilla_s3 = close_roll - 1.125 * hl_range
    
    # 4h volume average (20-period) for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34, 1)  # volume20, ema_34_1d, and +1 for roll
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_pp[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_34_1d_aligned[i]
        pp = camarilla_pp[i]
        r3 = camarilla_r3[i]
        s3 = camarilla_s3[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above R3 AND price > 1d EMA34 AND volume spike
            if price > r3 and price > ema_val and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: price breaks below S3 AND price < 1d EMA34 AND volume spike
            elif price < s3 and price < ema_val and volume[i] > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: price reverts to pivot point (PP)
            if position == 1 and price < pp:
                exit_signal = True
            elif position == -1 and price > pp:
                exit_signal = True
            
            # ATR-based trailing stop: 2.5 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_PPExit_ATRTrailingStop"
timeframe = "4h"
leverage = 1.0