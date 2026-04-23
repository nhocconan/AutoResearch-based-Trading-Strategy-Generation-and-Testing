#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above 4h Camarilla R3 AND 1d EMA34 rising AND volume > 1.5x 20-period average.
Short when price breaks below 4h Camarilla S3 AND 1d EMA34 falling AND volume > 1.5x 20-period average.
Exit when price retouches 4h Camarilla P (pivot) or ATR stoploss hit (2.0*ATR).
Uses discrete position sizing (0.25) to minimize fee churn and control drawdown.
Designed for 4h timeframe to target 20-50 trades/year per symbol (80-200 total over 4 years).
Works in both bull and bear markets by trading with the 1d trend and using volume confirmation to filter false breakouts.
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
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = prices.index.hour
    
    # Calculate 4h Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels: based on previous day's OHLC
    camarilla_H = high_4h
    camarilla_L = low_4h
    camarilla_C = close_4h
    camarilla_range = camarilla_H - camarilla_L
    
    camarilla_P = (camarilla_H + camarilla_L + camarilla_C) / 3.0
    camarilla_R3 = camarilla_C + (camarilla_range * 1.1 / 4.0)
    camarilla_S3 = camarilla_C - (camarilla_range * 1.1 / 4.0)
    
    # Align Camarilla levels to 4h timeframe (already on 4h, but align for consistency)
    camarilla_P_aligned = align_htf_to_ltf(prices, df_4h, camarilla_P)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_S3)
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_34_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_34)
    
    # EMA slope (rising/falling)
    ema_slope = np.zeros_like(ema_1d_34_aligned)
    ema_slope[1:] = ema_1d_34_aligned[1:] - ema_1d_34_aligned[:-1]
    
    # Volume average (20-period)
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
    start_idx = max(100, 2, 34, 20, 14, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_P_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or 
            np.isnan(camarilla_S3_aligned[i]) or np.isnan(ema_1d_34_aligned[i]) or 
            np.isnan(ema_slope[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
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
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        P = camarilla_P_aligned[i]
        R3 = camarilla_R3_aligned[i]
        S3 = camarilla_S3_aligned[i]
        ema_slope_val = ema_slope[i]
        
        if position == 0:
            # Long: Price breaks above Camarilla R3 AND 1d EMA34 rising AND volume spike
            if (price > R3 and 
                ema_slope_val > 0 and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Price breaks below Camarilla S3 AND 1d EMA34 falling AND volume spike
            elif (price < S3 and 
                  ema_slope_val < 0 and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retouches Camarilla P (pivot)
            if position == 1 and price <= P:
                exit_signal = True
            elif position == -1 and price >= P:
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
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R3S3_1dEMA34_Trend_VolumeConfirmation_ATRStop"
timeframe = "4h"
leverage = 1.0