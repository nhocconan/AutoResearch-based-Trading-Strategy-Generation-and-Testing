#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above 6h Camarilla R3 AND 1d EMA50 is rising AND volume > 1.3x 20-period average.
Short when price breaks below 6h Camarilla S3 AND 1d EMA50 is falling AND volume > 1.3x 20-period average.
Exit when price retouches 6h Camarilla H3/L3 levels or ATR stoploss hit (2.0*ATR).
Uses discrete position sizing (0.25) to minimize fee churn and control drawdown.
Designed for 6h timeframe to target 12-37 trades/year per symbol (50-150 total over 4 years).
Works in both bull and bear markets by trading with the 1d trend and using volume confirmation to filter false breakouts.
Camarilla R3/S3 levels provide institutional breakout/breakdown points with built-in retest levels (H3/L3).
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
    
    # Camarilla levels: based on previous 6h bar's range
    camarilla_h5 = np.zeros_like(close_6h)
    camarilla_h4 = np.zeros_like(close_6h)
    camarilla_h3 = np.zeros_like(close_6h)
    camarilla_l3 = np.zeros_like(close_6h)
    camarilla_l4 = np.zeros_like(close_6h)
    camarilla_l5 = np.zeros_like(close_6h)
    
    # Calculate for each 6h bar (starting from index 1 as we need previous bar)
    for i in range(1, len(close_6h)):
        prev_high = high_6h[i-1]
        prev_low = low_6h[i-1]
        prev_close = close_6h[i-1]
        range_val = prev_high - prev_low
        
        camarilla_h5[i] = prev_close + 1.1 * range_val * 1.1
        camarilla_h4[i] = prev_close + 1.1 * range_val * 0.55
        camarilla_h3[i] = prev_close + 1.1 * range_val * 0.275
        camarilla_l3[i] = prev_close - 1.1 * range_val * 0.275
        camarilla_l4[i] = prev_close - 1.1 * range_val * 0.55
        camarilla_l5[i] = prev_close - 1.1 * range_val * 1.1
    
    # Align Camarilla levels to 6h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_6h, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_6h, camarilla_l4)
    
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
    start_idx = max(100, 2, 50, 20, 14, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema_1d_50_aligned[i]) or np.isnan(ema_slope[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
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
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        h4 = camarilla_h4_aligned[i]
        l4 = camarilla_l4_aligned[i]
        ema_slope_val = ema_slope[i]
        
        if position == 0:
            # Long: Price breaks above Camarilla H3 AND 1d EMA50 rising AND volume spike
            if (price > h3 and 
                ema_slope_val > 0 and 
                volume[i] > 1.3 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Price breaks below Camarilla L3 AND 1d EMA50 falling AND volume spike
            elif (price < l3 and 
                  ema_slope_val < 0 and 
                  volume[i] > 1.3 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retouches Camarilla H4/L4 levels
            if position == 1 and price <= h4:
                exit_signal = True
            elif position == -1 and price >= l4:
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

name = "6H_Camarilla_R3S3_1dEMA50_Trend_VolumeConfirmation_ATRStop"
timeframe = "6h"
leverage = 1.0