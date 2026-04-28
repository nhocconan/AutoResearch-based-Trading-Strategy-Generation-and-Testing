#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_Volume
Hypothesis: 1-hour breakouts at Camarilla R1/S1 levels with 4-hour trend filter and volume confirmation. Uses 4h trend for direction (reduces whipsaw) and 1h for precise entry timing. Volume surge confirms breakout strength. Session filter (08-20 UTC) reduces noise. Targets 15-35 trades/year by requiring confluence of trend, level break, and volume. Works in bull/bear markets by following 4h trend.
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
    
    # Get 4-hour data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 4h EMA34 for trend filter
    ema_34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align all higher timeframe data to 1h
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Trend filter: price > EMA34 = bullish, < EMA34 = bearish
    trend_up = close > ema_34_4h_aligned
    trend_down = close < ema_34_4h_aligned
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 2.0)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Require session filter
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        # Entry conditions with trend alignment and volume surge
        # Long: price breaks above Camarilla R1 + 4h uptrend + volume surge
        long_entry = (close[i] > camarilla_r1_aligned[i] and 
                     trend_up[i] and 
                     volume_surge[i])
        
        # Short: price breaks below Camarilla S1 + 4h downtrend + volume surge
        short_entry = (close[i] < camarilla_s1_aligned[i] and 
                      trend_down[i] and 
                      volume_surge[i])
        
        # Exit on opposite level break with volume surge
        long_exit = close[i] < camarilla_s1_aligned[i] and volume_surge[i]
        short_exit = close[i] > camarilla_r1_aligned[i] and volume_surge[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.20
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.20  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.20   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0