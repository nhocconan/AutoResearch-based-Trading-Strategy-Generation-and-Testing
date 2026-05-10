#!/usr/bin/env python3
# 12h_Daily_Price_Action_Reversal
# Hypothesis: Daily price action reversal patterns (hammer/shooting star) at 12h timeframe with volume confirmation.
# Works in both bull and bear markets by capturing exhaustion moves after extended trends.
# Target: 12-30 trades/year (~48-120 total over 4 years) to stay within optimal trade frequency for 12h.

name = "12h_Daily_Price_Action_Reversal"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily candlestick patterns
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Hammer pattern: small body, long lower shadow, little/no upper shadow
    body_1d = np.abs(close_1d - open_1d)
    lower_shadow_1d = np.minimum(open_1d, close_1d) - low_1d
    upper_shadow_1d = high_1d - np.maximum(open_1d, close_1d)
    
    # Hammer conditions: lower shadow >= 2*body, upper shadow <= 0.1*body
    hammer = (lower_shadow_1d >= 2.0 * body_1d) & (upper_shadow_1d <= 0.1 * body_1d) & (body_1d > 0)
    
    # Shooting star pattern: small body, long upper shadow, little/no lower shadow
    shooting_star = (upper_shadow_1d >= 2.0 * body_1d) & (lower_shadow_1d <= 0.1 * body_1d) & (body_1d > 0)
    
    # Align daily patterns to 12h timeframe
    hammer_aligned = align_htf_to_ltf(prices, df_1d, hammer.astype(float))
    shooting_star_aligned = align_htf_to_ltf(prices, df_1d, shooting_star.astype(float))
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(hammer_aligned[i]) or np.isnan(shooting_star_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Hammer pattern with volume confirmation
            if hammer_aligned[i] > 0.5 and volume_filter[i] and session_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Shooting star pattern with volume confirmation
            elif shooting_star_aligned[i] > 0.5 and volume_filter[i] and session_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Shooting star pattern appears or volume drops
            if shooting_star_aligned[i] > 0.5 or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Hammer pattern appears or volume drops
            if hammer_aligned[i] > 0.5 or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals