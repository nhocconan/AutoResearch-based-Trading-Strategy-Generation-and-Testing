#!/usr/bin/env python3
"""
12h_1w_donchian_breakout_v1
Strategy: 12h Donchian breakout with 1w trend filter
Timeframe: 12h
Leverage: 1.0
Hypothesis: Uses Donchian channel breakout on 12h with 1-week EMA50 trend filter. 
Goes long when price breaks above 20-period high in uptrend, short when breaks below 20-period low in downtrend.
Uses volume confirmation to avoid false breaks. Designed to capture trends while minimizing whipsaw.
Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_donchian_breakout_v1"
timeframe = "12h"
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
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 12h Donchian channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h volume average (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        vol_current = volume[i]
        
        # Trend filter
        uptrend = price_close > ema_50_1w_aligned[i]
        downtrend = price_close < ema_50_1w_aligned[i]
        
        # Volume confirmation (above average)
        vol_confirm = vol_current > vol_ma[i]
        
        # Donchian breakout conditions
        breakout_up = price_high > high_roll[i-1]  # Current high > previous period's high
        breakout_down = price_low < low_roll[i-1]  # Current low < previous period's low
        
        # Long: upward breakout in uptrend with volume
        long_signal = breakout_up and uptrend and vol_confirm
        
        # Short: downward breakout in downtrend with volume
        short_signal = breakout_down and downtrend and vol_confirm
        
        # Exit when price crosses back to the middle of the channel
        channel_mid = (high_roll[i] + low_roll[i]) / 2.0
        exit_long = position == 1 and price_close < channel_mid
        exit_short = position == -1 and price_close > channel_mid
        
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