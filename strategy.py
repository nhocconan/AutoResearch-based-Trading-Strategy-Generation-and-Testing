#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with weekly EMA34 trend filter and volume confirmation.
Long when price breaks above Camarilla R3 AND price > weekly EMA34 AND volume > 2.0x 20-period average.
Short when price breaks below Camarilla S3 AND price < weekly EMA34 AND volume > 2.0x 20-period average.
Exit when price reverts to Camarilla H5/L5 level OR ATR trailing stop (2.5*ATR from extreme).
Uses 1w HTF for trend alignment to capture major trend direction.
Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25.
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
    
    # Calculate weekly EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need enough for EMA
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla pivot levels from previous day to avoid look-ahead
    # Roll by 96 periods (24h * 4 = 96 six-hour bars in a day)
    high_roll = np.roll(high, 96)
    low_roll = np.roll(low, 96)
    close_roll = np.roll(close, 96)
    high_roll[:96] = np.nan
    low_roll[:96] = np.nan
    close_roll[:96] = np.nan
    
    pivot = (high_roll + low_roll + close_roll) / 3.0
    range_hl = high_roll - low_roll
    
    camarilla_h5 = pivot + (range_hl * 1.1 / 2)
    camarilla_l5 = pivot - (range_hl * 1.1 / 2)
    camarilla_h3 = pivot + (range_hl * 1.1 / 4)
    camarilla_l3 = pivot - (range_hl * 1.1 / 4)
    camarilla_h4 = pivot + (range_hl * 1.1 / 3)
    camarilla_l4 = pivot - (range_hl * 1.1 / 3)
    
    # 6h volume average (20-period) for spike filter
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
    start_idx = max(96, 34, 20)  # camarilla (96 roll), ema_34_1w, vol_ma
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or np.isnan(camarilla_h5[i]) or np.isnan(camarilla_l5[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_34_1w_aligned[i]
        h3 = camarilla_h3[i]
        l3 = camarilla_l3[i]
        h5 = camarilla_h5[i]
        l5 = camarilla_l5[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above H3 AND price > weekly EMA34 AND volume spike
            if price > h3 and price > ema_val and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: price breaks below L3 AND price < weekly EMA34 AND volume spike
            elif price < l3 and price < ema_val and volume[i] > 2.0 * vol_ma_val:
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
            
            # Primary exit: price reverts to H5/L5 levels
            if position == 1 and price < h5:
                exit_signal = True
            elif position == -1 and price > l5:
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

name = "6H_Camarilla_R3S3_Breakout_1wEMA34_Trend_VolumeSpike_H5L5Exit_ATRTrailingStop"
timeframe = "6h"
leverage = 1.0