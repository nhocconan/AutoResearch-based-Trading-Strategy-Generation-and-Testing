#!/usr/bin/env python3
"""
1h_4h_1d_camarilla_pivot_volume_v1
Strategy: 1h Camarilla pivot breakout with volume confirmation and 4h/1d trend filter
Timeframe: 1h
Leverage: 1.0
Hypothesis: Uses 1h Camarilla pivot levels (resistance/support) for breakout entries with volume confirmation (>1.5x average volume) and filtered by 4h EMA20 and 1d EMA50 trend alignment. Designed to capture breakouts in trending markets while avoiding false breakouts in chop. Uses higher timeframes for direction (4h/1d) and 1h only for timing. Target: 60-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_camarilla_pivot_volume_v1"
timeframe = "1h"
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
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # 1h Camarilla pivot levels (based on previous day's OHLC)
    # We'll use daily high/low/close from 1d data
    # Camarilla levels: H4 = Close + 1.1*(High-Low)/2, L4 = Close - 1.1*(High-Low)/2
    # We'll use these as breakout levels
    
    # 1h EMA20 for trend filter
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 4h EMA20 for trend filter
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    # Calculate Camarilla levels from previous day's data
    # We need to get the previous day's high, low, close for each 1h bar
    # Since we have 1d data, we can use the previous day's values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_for_cama = df_1d['close'].values  # same as close_1d above
    
    # Calculate Camarilla levels for each day
    camarilla_H4 = close_1d_for_cama + 1.1 * (high_1d - low_1d) / 2
    camarilla_L4 = close_1d_for_cama - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 1h timeframe
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_20[i]) or np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg[i]) or
            np.isnan(camarilla_H4_aligned[i]) or np.isnan(camarilla_L4_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        price_close = close[i]
        
        # Trend filters: price above/both EMAs for long, below/both for short
        uptrend_4h = price_close > ema_20_4h_aligned[i]
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        downtrend_4h = price_close < ema_20_4h_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        
        # Breakout conditions using Camarilla levels
        breakout_up = price_close > camarilla_H4_aligned[i]
        breakout_down = price_close < camarilla_L4_aligned[i]
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: upward breakout with volume in uptrend (both 4h and 1d)
        long_signal = breakout_up and vol_confirmed and uptrend_4h and uptrend_1d
        
        # Short: downward breakout with volume in downtrend (both 4h and 1d)
        short_signal = breakout_down and vol_confirmed and downtrend_4h and downtrend_1d
        
        # Exit when price returns to the EMA20 (1h) or opposite Camarilla level
        exit_long = position == 1 and (price_close < ema_20[i] or price_close < camarilla_L4_aligned[i])
        exit_short = position == -1 and (price_close > ema_20[i] or price_close > camarilla_H4_aligned[i])
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals