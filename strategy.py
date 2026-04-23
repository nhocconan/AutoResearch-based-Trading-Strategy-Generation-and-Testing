#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above 4h Camarilla R3 AND 1d EMA50 is rising AND volume > 1.5x 20-period average.
Short when price breaks below 4h Camarilla S3 AND 1d EMA50 is falling AND volume > 1.5x 20-period average.
Exit when price retouches 4h Camarilla pivot point or ATR stoploss hit (2.0*ATR).
Uses discrete position sizing (0.20) to minimize fee churn and control drawdown.
Designed for 1h timeframe to target 15-37 trades/year per symbol (60-150 total over 4 years).
Works in both bull and bear markets by trading with the 1d trend and using volume confirmation to filter false breakouts.
4h Camarilla levels provide institutional support/resistance; 1d EMA50 filters counter-trend moves.
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
    
    # Calculate 4h Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels (based on previous day's OHLC)
    camarilla_h4 = (high_4h + low_4h + close_4h) / 3.0
    camarilla_l4 = (high_4h + low_4h + close_4h) / 3.0
    camarilla_range = high_4h - low_4h
    
    camarilla_r3 = camarilla_h4 + camarilla_range * 1.1 / 4.0
    camarilla_s3 = camarilla_l4 - camarilla_range * 1.1 / 4.0
    camarilla_pivot = camarilla_h4  # Pivot point
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pivot)
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # EMA slope (rising/falling)
    ema_slope = np.zeros_like(ema_1d_50_aligned)
    ema_slope[1:] = ema_1d_50_aligned[1:] - ema_1d_50_aligned[:-1]
    
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
    start_idx = max(100, 5, 50, 20, 14, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(ema_1d_50_aligned[i]) or 
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
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        pivot = camarilla_pivot_aligned[i]
        ema_slope_val = ema_slope[i]
        
        if position == 0:
            # Long: Price breaks above Camarilla R3 AND 1d EMA50 rising AND volume spike
            if (price > r3 and 
                ema_slope_val > 0 and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short: Price breaks below Camarilla S3 AND 1d EMA50 falling AND volume spike
            elif (price < s3 and 
                  ema_slope_val < 0 and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.20
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retouches Camarilla pivot point
            if position == 1 and price <= pivot:
                exit_signal = True
            elif position == -1 and price >= pivot:
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

name = "1H_Camarilla_R3S3_1dEMA50_Trend_VolumeConfirmation_ATRStop"
timeframe = "1h"
leverage = 1.0