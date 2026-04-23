#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h Camarilla R3/S3 breakout with volume confirmation and ATR stoploss.
Long when price breaks above 4h Camarilla R3 AND volume > 1.5x 20-period average.
Short when price breaks below 4h Camarilla S3 AND volume > 1.5x 20-period average.
Exit when price retouches 4h Camarilla H3/L3 levels or ATR stoploss hit (2.0*ATR).
Uses discrete position sizing (0.20) to control drawdown and fee drag.
Designed for 1h timeframe to target 15-37 trades/year per symbol (60-150 total over 4 years).
Uses 4h for signal direction, 1h only for entry timing. Session filter (08-20 UTC) reduces noise.
Works in both bull and bear markets by using volume confirmation to filter false breakouts.
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
    
    # Pre-compute session hours (08-20 UTC) to avoid per-bar datetime ops
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h Camarilla levels (based on previous 4h bar's OHLC)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels (based on previous period's OHLC)
    camarilla_h3 = (high_4h + low_4h + close_4h) / 3.0
    camarilla_l3 = (high_4h + low_4h + close_4h) / 3.0
    camarilla_range = high_4h - low_4h
    
    camarilla_r3 = camarilla_h3 + camarilla_range * 1.1 / 4.0
    camarilla_s3 = camarilla_l3 - camarilla_range * 1.1 / 4.0
    camarilla_h3_level = camarilla_h3 + camarilla_range * 1.1 / 2.0
    camarilla_l3_level = camarilla_l3 - camarilla_range * 1.1 / 2.0
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3_level)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3_level)
    
    # Volume average (20-period) on 1h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        
        if position == 0:
            # Long: Price breaks above 4h Camarilla R3 AND volume spike
            if (price > r3 and volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short: Price breaks below 4h Camarilla S3 AND volume spike
            elif (price < s3 and volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.20
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retouches 4h Camarilla H3/L3 levels
            if position == 1 and price <= h3:
                exit_signal = True
            elif position == -1 and price >= l3:
                exit_signal = True
            
            # ATR-based stoploss: 2.0 * ATR from entry
            if position == 1 and price < entry_price - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > entry_price + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R3S3_VolumeConfirmation_ATRStop"
timeframe = "1h"
leverage = 1.0