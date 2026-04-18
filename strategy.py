#!/usr/bin/env python3
"""
12h Daily Range Breakout with Volume Spike and EMA Trend Filter
Hypothesis: The previous day's high and low act as key support/resistance levels.
Breakouts beyond these levels with volume confirmation and aligned with 1-day EMA trend
capture momentum moves. Works in both bull and bear markets by requiring volume confirmation
to avoid false breakouts and using EMA trend to align with higher timeframe direction.
Designed for 12-37 trades/year on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for previous day's high/low and EMA (once before loop)
    df_d = get_htf_data(prices, '1d')
    
    # Previous day's high and low (shifted by 1 to avoid look-ahead)
    prev_high = df_d['high'].shift(1).values
    prev_low = df_d['low'].shift(1).values
    
    # 1-day EMA34 for trend filter
    ema_34 = pd.Series(df_d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 12h timeframe
    prev_high_aligned = align_htf_to_ltf(prices, df_d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_d, prev_low)
    ema_34_aligned = align_htf_to_ltf(prices, df_d, ema_34)
    
    # Volume spike: 2x 20-period average on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(prev_high_aligned[i]) or 
            np.isnan(prev_low_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ph = prev_high_aligned[i]
        pl = prev_low_aligned[i]
        ema_trend = ema_34_aligned[i]
        
        if position == 0:
            # Long: break above previous day's high with volume spike and bullish EMA trend
            if price > ph and volume_spike[i] and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: break below previous day's low with volume spike and bearish EMA trend
            elif price < pl and volume_spike[i] and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: price returns to previous day's low or price breaks below EMA
            if price <= pl or price < ema_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: price returns to previous day's high or price breaks above EMA
            if price >= ph or price > ema_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_DailyRange_Breakout_Volume_EMA"
timeframe = "12h"
leverage = 1.0