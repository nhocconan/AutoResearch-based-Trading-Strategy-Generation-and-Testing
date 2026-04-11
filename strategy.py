#!/usr/bin/env python3
"""
12h_1d_camarilla_pivot_volume_trend_v1
Strategy: 12h Camarilla pivot breakout with volume confirmation and 1d EMA trend filter
Timeframe: 12h
Leverage: 1.0
Hypothesis: Uses 12h Camarilla pivot levels (based on 1d OHLC) for breakout entries with volume confirmation (>1.8x average volume) and filtered by 1d EMA50 trend alignment. Designed to capture breakouts in trending markets while avoiding false breakouts in chop. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_pivot_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h EMA20 for trend filter (optional, can be removed if too restrictive)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.8 * vol_avg)  # Volume spike filter
    
    # Calculate Camarilla levels from previous day's data using 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_for_cama = df_1d['close'].values
    
    # Calculate Camarilla levels (H4 and L4)
    camarilla_H4 = close_1d_for_cama + 1.1 * (high_1d - low_1d) / 2
    camarilla_L4 = close_1d_for_cama - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_20[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_avg[i]) or
            np.isnan(camarilla_H4_aligned[i]) or np.isnan(camarilla_L4_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        
        # Breakout conditions using Camarilla levels
        breakout_up = price_close > camarilla_H4_aligned[i]
        breakout_down = price_close < camarilla_L4_aligned[i]
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: upward breakout with volume in uptrend
        long_signal = breakout_up and vol_confirmed and uptrend_1d
        
        # Short: downward breakout with volume in downtrend
        short_signal = breakout_down and vol_confirmed and downtrend_1d
        
        # Exit when price returns to the 12h EMA20 or opposite Camarilla level
        exit_long = position == 1 and (price_close < ema_20[i] or price_close < camarilla_L4_aligned[i])
        exit_short = position == -1 and (price_close > ema_20[i] or price_close > camarilla_H4_aligned[i])
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals