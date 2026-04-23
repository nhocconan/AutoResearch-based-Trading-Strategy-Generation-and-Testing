#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R4/S4 breakout with 1d EMA34 trend filter and volume spike confirmation.
Long when price breaks above R4 AND daily close > daily EMA34 AND volume > 2.0x average.
Short when price breaks below S4 AND daily close < daily EMA34 AND volume > 2.0x average.
Exit when price reaches opposite Camarilla level (R4 for long exit, S4 for short exit) or after 12 bars max hold.
Uses discrete position sizing (0.30) to balance profit and risk. Targets 20-40 trades/year per symbol.
Camarilla levels provide precise support/resistance, daily trend filter avoids counter-trend trades,
volume spike confirms institutional interest. Max hold prevents stagnation.
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
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA34 on 1d data
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla uses previous day's high, low, close
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = close_1d[0]  # first bar uses current bar
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    # Camarilla R4, S4 levels (more extreme than R3/S3)
    rangep = prev_high_1d - prev_low_1d
    camarilla_r4 = prev_close_1d + rangep * 1.1 / 2
    camarilla_s4 = prev_close_1d - rangep * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        daily_trend_up = close[i] > ema34_1d_aligned[i]
        daily_trend_down = close[i] < ema34_1d_aligned[i]
        
        vol_current = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            bars_since_entry = 0
            # Long: Price breaks above R4 AND daily uptrend AND volume spike
            if (high[i] > camarilla_r4_aligned[i] and daily_trend_up and 
                vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.30
                position = 1
            # Short: Price breaks below S4 AND daily downtrend AND volume spike
            elif (low[i] < camarilla_s4_aligned[i] and daily_trend_down and 
                  vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.30
                position = -1
        else:
            bars_since_entry += 1
            
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price reaches S4 (opposite level) OR max hold reached (12 bars = 2 days)
                if low[i] < camarilla_s4_aligned[i] or bars_since_entry >= 12:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price reaches R4 (opposite level) OR max hold reached
                if high[i] > camarilla_r4_aligned[i] or bars_since_entry >= 12:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "4H_Camarilla_R4S4_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0