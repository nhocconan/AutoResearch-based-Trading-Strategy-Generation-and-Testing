#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and choppiness regime filter.
Long when price breaks above Camarilla R3 AND 1d volume > 2.0x 20-period average AND choppiness > 61.8 (range).
Short when price breaks below Camarilla S3 AND 1d volume > 2.0x 20-period average AND choppiness > 61.8 (range).
Exit when price retouches Camarilla pivot point (PP) or ATR stoploss hit (1.5*ATR).
Uses discrete position sizing (0.25) to minimize fee churn and control drawdown.
Targets 12-37 trades/year per symbol (50-150 total over 4 years) by using 1d HTF structure and range regime filter.
Designed to work in both bull and bear markets by fading extremes in ranging markets and using tight risk control.
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
    open_time = prices['open_time'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Calculate 1d Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels (based on previous day's OHLC)
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    camarilla_r3 = camarilla_pp + (high_1d - low_1d) * 1.1 / 4.0
    camarilla_s3 = camarilla_pp - (high_1d - low_1d) * 1.1 / 4.0
    
    # Align Camarilla levels to 12h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 1d ATR(14) for stoploss
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]  # first bar
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 1d volume average (20-period)
    vol_ma_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 12h choppiness index (14-period) for regime filter
    tr1_12h = np.abs(high - low)
    tr2_12h = np.abs(high - np.roll(close, 1))
    tr3_12h = np.abs(low - np.roll(close, 1))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    tr_12h[0] = tr1_12h[0]  # first bar
    atr_sum = pd.Series(tr_12h).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / np.log10(14) / (max_high - min_low))
    chop = np.where((max_high - min_low) != 0, chop, 50.0)  # avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(100, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only
        if hours[i] < 8 or hours[i] > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma_1d_aligned[i]
        atr_val = atr_1d_aligned[i]
        pp = camarilla_pp_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        chop_val = chop[i]
        
        if position == 0:
            # Long: Price breaks above Camarilla R3 AND volume spike AND choppy market (range)
            if (price > r3 and 
                volume[i] > 2.0 * vol_ma_val and 
                chop_val > 61.8):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Price breaks below Camarilla S3 AND volume spike AND choppy market (range)
            elif (price < s3 and 
                  volume[i] > 2.0 * vol_ma_val and 
                  chop_val > 61.8):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retouches Camarilla pivot point
            if position == 1 and price <= pp:
                exit_signal = True
            elif position == -1 and price >= pp:
                exit_signal = True
            
            # ATR-based stoploss: 1.5 * ATR from entry
            if position == 1 and price < entry_price - 1.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > entry_price + 1.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_R3S3_1dVolumeSpike_Chop618_ATRStop"
timeframe = "12h"
leverage = 1.0