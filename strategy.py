#!/usr/bin/env python3
"""
6h_12h_1d_camarilla_pivot_volume_v1
Strategy: 6h Camarilla pivot breakout with volume confirmation and 12h/1d trend filter
Timeframe: 6h
Leverage: 1.0
Hypothesis: Uses 6h Camarilla pivot levels (based on prior 12h candles) for breakout entries with volume confirmation (>1.5x average volume) and filtered by 12h and 1d EMA20 trend alignment. Designed to capture breakouts in trending markets while avoiding false breakouts in chop. Uses higher timeframes for direction (12h/1d) and 6h only for timing. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_camarilla_pivot_volume_v1"
timeframe = "6h"
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
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # 6h EMA20 for trend filter
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 12h EMA20 for trend filter
    close_12h = df_12h['close'].values
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # 1d EMA20 for trend filter
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    # Calculate Camarilla levels from previous 12h period's OHLC
    # We'll use 12h high/low/close from 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h_for_cama = df_12h['close'].values
    
    # Calculate Camarilla levels for each 12h period
    camarilla_H4 = close_12h_for_cama + 1.1 * (high_12h - low_12h) / 2
    camarilla_L4 = close_12h_for_cama - 1.1 * (high_12h - low_12h) / 2
    
    # Align Camarilla levels to 6h timeframe
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_H4)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_L4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_20[i]) or np.isnan(ema_20_12h_aligned[i]) or 
            np.isnan(ema_20_1d_aligned[i]) or np.isnan(vol_avg[i]) or
            np.isnan(camarilla_H4_aligned[i]) or np.isnan(camarilla_L4_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filters: price above/both EMAs for long, below/both for short
        uptrend_12h = price_close > ema_20_12h_aligned[i]
        uptrend_1d = price_close > ema_20_1d_aligned[i]
        downtrend_12h = price_close < ema_20_12h_aligned[i]
        downtrend_1d = price_close < ema_20_1d_aligned[i]
        
        # Breakout conditions using Camarilla levels
        breakout_up = price_close > camarilla_H4_aligned[i]
        breakout_down = price_close < camarilla_L4_aligned[i]
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: upward breakout with volume in uptrend (both 12h and 1d)
        long_signal = breakout_up and vol_confirmed and uptrend_12h and uptrend_1d
        
        # Short: downward breakout with volume in downtrend (both 12h and 1d)
        short_signal = breakout_down and vol_confirmed and downtrend_12h and downtrend_1d
        
        # Exit when price returns to the EMA20 (6h) or opposite Camarilla level
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