#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 reversal with 12h EMA50 trend filter and volume confirmation.
Long when price <= S3 AND 12h EMA50 rising AND volume > 1.3x 20-period MA.
Short when price >= R3 AND 12h EMA50 falling AND volume > 1.3x 20-period MA.
Exit when price crosses the daily pivot (PP) or EMA50 reverses.
Uses 12h HTF for trend filter to avoid counter-trend trades, volume spike for momentum confirmation.
Camarilla levels from 1d provide precise intraday support/resistance that works in both trending and ranging markets.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
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
    
    # Calculate 12h EMA50 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 1d Camarilla levels (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_R4 = np.zeros(len(df_1d))
    camarilla_R3 = np.zeros(len(df_1d))
    camarilla_S3 = np.zeros(len(df_1d))
    camarilla_S4 = np.zeros(len(df_1d))
    camarilla_PP = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        if i == 0:
            camarilla_R4[i] = np.nan
            camarilla_R3[i] = np.nan
            camarilla_S3[i] = np.nan
            camarilla_S4[i] = np.nan
            camarilla_PP[i] = np.nan
            continue
            
        # Use previous day's OHLC for today's Camarilla levels
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        
        camarilla_PP[i] = (prev_high + prev_low + prev_close) / 3
        range_val = prev_high - prev_low
        camarilla_R3[i] = camarilla_PP[i] + range_val * 1.1 / 4
        camarilla_S3[i] = camarilla_PP[i] - range_val * 1.1 / 4
        camarilla_R4[i] = camarilla_PP[i] + range_val * 1.1 / 2
        camarilla_S4[i] = camarilla_PP[i] - range_val * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    camarilla_PP_aligned = align_htf_to_ltf(prices, df_1d, camarilla_PP)
    
    # Calculate 6h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or 
            np.isnan(camarilla_S3_aligned[i]) or np.isnan(camarilla_PP_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate EMA50 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_50_12h_aligned[i-1]
            ema_rising = ema_50_12h_aligned[i] > ema_prev
            ema_falling = ema_50_12h_aligned[i] < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 6h volume > 1.3x 20-period MA (moderate threshold to balance frequency)
        vol_filter = volume[i] > 1.3 * vol_ma_20[i]
        
        if position == 0:
            # Long: price <= S3 AND EMA50 rising AND volume filter
            if close[i] <= camarilla_S3_aligned[i] and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price >= R3 AND EMA50 falling AND volume filter
            elif close[i] >= camarilla_R3_aligned[i] and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price crosses above daily pivot OR EMA50 starts falling
                if close[i] > camarilla_PP_aligned[i] or (i >= start_idx + 1 and ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price crosses below daily pivot OR EMA50 starts rising
                if close[i] < camarilla_PP_aligned[i] or (i >= start_idx + 1 and ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R3S3_Reversal_12hEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0