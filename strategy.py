#!/usr/bin/env python3
"""
1h_4h1d_Camarilla_Pivot_Bounce
Hypothesis: Mean reversion at 4h/1d Camarilla pivots with 1h entry timing. Uses 4h/1d pivot levels as institutional support/resistance. In ranging markets (common in 2025-2026 BTC/ETH), price tends to revert to these levels. Volume confirmation filters false breaks. Session filter (08-20 UTC) reduces noise. Targets 15-30 trades/year by requiring confluence of level, volume, and session.
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
    
    # Get 4h data for pivot calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels from previous 4h bar
    prev_close_4h = df_4h['close'].shift(1).values
    prev_high_4h = df_4h['high'].shift(1).values
    prev_low_4h = df_4h['low'].shift(1).values
    camarilla_r4_4h = prev_close_4h + (prev_high_4h - prev_low_4h) * 1.1 / 2  # R4
    camarilla_r3_4h = prev_close_4h + (prev_high_4h - prev_low_4h) * 1.1 / 4  # R3
    camarilla_s3_4h = prev_close_4h - (prev_high_4h - prev_low_4h) * 1.1 / 4  # S3
    camarilla_s4_4h = prev_close_4h - (prev_high_4h - prev_low_4h) * 1.1 / 2  # S4
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all higher timeframe data to 1h
    camarilla_r4_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r4_4h)
    camarilla_r3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h)
    camarilla_s3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h)
    camarilla_s4_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s4_4h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5x 24-period average
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_surge = volume > (vol_ma_24 * 1.5)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r4_4h_aligned[i]) or np.isnan(camarilla_r3_4h_aligned[i]) or
            np.isnan(camarilla_s3_4h_aligned[i]) or np.isnan(camarilla_s4_4h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_surge[i]) or np.isnan(in_session[i])):
            signals[i] = 0.0
            continue
        
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Entry conditions: bounce off S3/S4 in uptrend, R3/R4 in downtrend
        long_entry = ((close[i] <= camarilla_s3_4h_aligned[i] or close[i] <= camarilla_s4_4h_aligned[i]) and
                      ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and  # rising EMA = uptrend
                      volume_surge[i])
        
        short_entry = ((close[i] >= camarilla_r3_4h_aligned[i] or close[i] >= camarilla_r4_4h_aligned[i]) and
                       ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and  # falling EMA = downtrend
                       volume_surge[i])
        
        # Exit when price moves back toward midpoint
        midpoint = (camarilla_r4_4h_aligned[i] + camarilla_s4_4h_aligned[i]) / 2
        long_exit = close[i] >= midpoint and position == 1
        short_exit = close[i] <= midpoint and position == -1
        
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

name = "1h_4h1d_Camarilla_Pivot_Bounce"
timeframe = "1h"
leverage = 1.0