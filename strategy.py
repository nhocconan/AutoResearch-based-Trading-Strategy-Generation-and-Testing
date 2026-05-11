#!/usr/bin/env python3
"""
1h_4h_1d_Camarilla_Breakout_with_Volume_and_Trend
Hypothesis: Uses 1d trend (price above/below EMA50) as directional filter, 
4h Camarilla pivot levels (R1/S1) for breakout entries, and volume confirmation.
Trades only during 08-20 UTC session to avoid low-liquidity hours.
Designed for low trade frequency (15-37/year) by requiring multiple confluences.
Works in bull/bear markets by following higher-timeframe trend direction.
"""

name = "1h_4h_1d_Camarilla_Breakout_with_Volume_and_Trend"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # 1h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # --- 1d EMA50 for Trend Filter ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_1h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # --- 4h OHLC for Camarilla Pivots ---
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 4h bar
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_multiplier = 1.1 / 12
    r1_4h = close_4h + (high_4h - low_4h) * camarilla_multiplier
    s1_4h = close_4h - (high_4h - low_4h) * camarilla_multiplier
    
    # Align to 1h (previous 4h bar close)
    r1_4h_1h = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_1h = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # --- 1h Volume Spike (20-bar average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_1h[i]) or np.isnan(r1_4h_1h[i]) or 
            np.isnan(s1_4h_1h[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position == 1:
                signals[i] = 0.0
                position = 0
            elif position == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend direction from 1d EMA50
        uptrend = close[i] > ema_50_1d_1h[i]
        downtrend = close[i] < ema_50_1d_1h[i]
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: uptrend + break above R1 + volume
            if (uptrend and 
                close[i] > r1_4h_1h[i] and 
                volume_spike):
                signals[i] = 0.20
                position = 1
            # Short: downtrend + break below S1 + volume
            elif (downtrend and 
                  close[i] < s1_4h_1h[i] and 
                  volume_spike):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions: trend reversal or opposite breakout
            if position == 1:
                # Exit long: downtrend OR break below S1
                if downtrend or close[i] < s1_4h_1h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: uptrend OR break above R1
                if uptrend or close[i] > r1_4h_1h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals