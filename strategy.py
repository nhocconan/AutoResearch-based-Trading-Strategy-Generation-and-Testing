#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA34 trend filter and volume confirmation.
Long when price >= R1 AND 4h EMA34 rising AND volume > 1.5x 20-period MA.
Short when price <= S1 AND 4h EMA34 falling AND volume > 1.5x 20-period MA.
Exit when price crosses the daily pivot (PP) or EMA34 reverses.
Uses 4h HTF for trend filter to avoid counter-trend trades, volume spike for momentum confirmation.
Camarilla levels from 1d provide precise intraday support/resistance that works in both trending and ranging markets.
Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
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
    
    # Calculate 4h EMA34 for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 1d Camarilla levels (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_R1 = np.zeros(len(df_1d))
    camarilla_S1 = np.zeros(len(df_1d))
    camarilla_PP = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        if i == 0:
            camarilla_R1[i] = np.nan
            camarilla_S1[i] = np.nan
            camarilla_PP[i] = np.nan
            continue
            
        # Use previous day's OHLC for today's Camarilla levels
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        
        camarilla_PP[i] = (prev_high + prev_low + prev_close) / 3
        range_val = prev_high - prev_low
        camarilla_R1[i] = camarilla_PP[i] + range_val * 1.1 / 12
        camarilla_S1[i] = camarilla_PP[i] - range_val * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    camarilla_PP_aligned = align_htf_to_ltf(prices, df_1d, camarilla_PP)
    
    # Calculate 1h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(camarilla_R1_aligned[i]) or 
            np.isnan(camarilla_S1_aligned[i]) or np.isnan(camarilla_PP_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_4h_aligned[i-1]
            ema_rising = ema_34_4h_aligned[i] > ema_prev
            ema_falling = ema_34_4h_aligned[i] < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 1h volume > 1.5x 20-period MA (strict threshold to reduce frequency)
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price >= R1 AND EMA34 rising AND volume filter
            if close[i] >= camarilla_R1_aligned[i] and ema_rising and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short: price <= S1 AND EMA34 falling AND volume filter
            elif close[i] <= camarilla_S1_aligned[i] and ema_falling and vol_filter:
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price crosses below daily pivot OR EMA34 starts falling
                if close[i] < camarilla_PP_aligned[i] or (i >= start_idx + 1 and ema_34_4h_aligned[i] < ema_34_4h_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price crosses above daily pivot OR EMA34 starts rising
                if close[i] > camarilla_PP_aligned[i] or (i >= start_idx + 1 and ema_34_4h_aligned[i] > ema_34_4h_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R1S1_Breakout_4hEMA34_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0