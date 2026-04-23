#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 1d trend filter and volume confirmation.
Long when price breaks above 6h Camarilla R3 AND 1d close > 1d open (bullish daily candle) AND volume > 1.5x 20-period average.
Short when price breaks below 6h Camarilla S3 AND 1d close < 1d open (bearish daily candle) AND volume > 1.5x 20-period average.
Exit when price retouches 6h Camarilla pivot point (PP) or ATR stoploss hit (2.0*ATR).
Uses discrete position sizing (0.25) to minimize fee churn and control drawdown.
Designed for 6h timeframe to target 12-37 trades/year per symbol (50-150 total over 4 years).
Works in both bull and bear markets by trading with daily bias and using volume confirmation to filter false breakouts.
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
    
    # Calculate 6h Camarilla pivot levels (based on prior 6h bar)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 2:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Camarilla levels: based on prior 6h bar's high, low, close
    camarilla_pp = (high_6h + low_6h + close_6h) / 3.0
    camarilla_r3 = camarilla_pp + (high_6h - low_6h) * 1.1 / 4.0
    camarilla_s3 = camarilla_pp - (high_6h - low_6h) * 1.1 / 4.0
    
    # Align Camarilla levels to 6h timeframe (already aligned by get_htf_data + align_htf_to_ltf)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_6h, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_s3)
    
    # Calculate 1d trend filter: bullish if close > open, bearish if close < open
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    open_1d = df_1d['open'].values
    close_1d = df_1d['close'].values
    daily_bullish = close_1d > open_1d  # True for bullish daily candle
    daily_bearish = close_1d < open_1d  # True for bearish daily candle
    
    # Align daily bias to 6h timeframe
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish.astype(float))
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish.astype(float))
    
    # Volume average (20-period) on 6h timeframe
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
    start_idx = max(100, 2, 1, 20, 14, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(daily_bullish_aligned[i]) or 
            np.isnan(daily_bearish_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
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
        pp = camarilla_pp_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        bullish = daily_bullish_aligned[i] > 0.5
        bearish = daily_bearish_aligned[i] > 0.5
        
        if position == 0:
            # Long: Price breaks above Camarilla R3 AND daily bullish AND volume spike
            if (price > r3 and 
                bullish and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Price breaks below Camarilla S3 AND daily bearish AND volume spike
            elif (price < s3 and 
                  bearish and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retouches Camarilla pivot point (PP)
            if position == 1 and price <= pp:
                exit_signal = True
            elif position == -1 and price >= pp:
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

name = "6H_Camarilla_R3S3_1dBias_VolumeConfirmation_ATRStop"
timeframe = "6h"
leverage = 1.0