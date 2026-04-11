#!/usr/bin/env python3
"""
1h_4h_1d_camarilla_breakout_volume_trend_v1
Strategy: 1h Camarilla breakout with volume confirmation and 4h/1d trend filter
Timeframe: 1h
Leverage: 1.0
Hypothesis: Uses 4h/1d trend direction to filter 1h breakouts from 1d Camarilla pivot levels.
             Volume confirmation (>2x average volume) reduces false breakouts.
             Designed to capture momentum moves in trending markets while avoiding chop.
             Target: 60-150 total trades over 4 years (15-37/year) by using higher timeframes for direction
             and 1h only for precise entry timing.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_camarilla_breakout_volume_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d EMA100 for trend filter (slower trend)
    close_1d = df_1d['close'].values
    ema_100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    # Volume average (24-period = 1 day at 1h)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (2.0 * vol_avg)  # Volume spike filter
    
    # Calculate 1d Camarilla levels from previous day's data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_for_cama = df_1d['close'].values
    
    # Calculate Camarilla levels (H4 and L4)
    camarilla_H4 = close_1d_for_cama + 1.1 * (high_1d - low_1d) / 2
    camarilla_L4 = close_1d_for_cama - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 1h timeframe
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_100_1d_aligned[i]) or 
            np.isnan(vol_avg[i]) or
            np.isnan(camarilla_H4_aligned[i]) or np.isnan(camarilla_L4_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        price_close = close[i]
        
        # Trend filter: both 4h and 1d must agree on direction
        uptrend = price_close > ema_50_4h_aligned[i] and price_close > ema_100_1d_aligned[i]
        downtrend = price_close < ema_50_4h_aligned[i] and price_close < ema_100_1d_aligned[i]
        
        # Breakout conditions using 1d Camarilla levels
        breakout_up = price_close > camarilla_H4_aligned[i]
        breakout_down = price_close < camarilla_L4_aligned[i]
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: upward breakout with volume in uptrend
        long_signal = breakout_up and vol_confirmed and uptrend
        
        # Short: downward breakout with volume in downtrend
        short_signal = breakout_down and vol_confirmed and downtrend
        
        # Exit when price returns to the 4h EMA50 or opposite Camarilla level
        exit_long = position == 1 and (price_close < ema_50_4h_aligned[i] or price_close < camarilla_L4_aligned[i])
        exit_short = position == -1 and (price_close > ema_50_4h_aligned[i] or price_close > camarilla_H4_aligned[i])
        
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