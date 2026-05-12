#!/usr/bin/env python3
"""
6h_RSI_Streak_With_Volume_Confirmation
Hypothesis: RSI streak (consecutive closes above/below previous close) identifies overextended moves. 
Combined with volume confirmation (1.5x average) and 1d trend filter (EMA50), it captures mean-reversion 
opportunities in both bull and bear markets. RSI streak >2 indicates exhaustion, while volume spike 
confirms participation. Works in ranging markets where streaks frequently occur.
"""

name = "6h_RSI_Streak_With_Volume_Confirmation"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI Streak: count consecutive closes > previous close (bullish streak) 
    # or < previous close (bearish streak)
    price_changes = np.diff(close, prepend=close[0])
    bullish_streak = np.where(price_changes > 0, 1, 0)
    bearish_streak = np.where(price_changes < 0, 1, 0)
    
    # Calculate consecutive streaks
    bull_streak_count = np.zeros(n)
    bear_streak_count = np.zeros(n)
    
    for i in range(1, n):
        if price_changes[i] > 0:
            bull_streak_count[i] = bull_streak_count[i-1] + 1
            bear_streak_count[i] = 0
        elif price_changes[i] < 0:
            bear_streak_count[i] = bear_streak_count[i-1] + 1
            bull_streak_count[i] = 0
        else:
            bull_streak_count[i] = 0
            bear_streak_count[i] = 0
    
    # Volume spike: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # 1d EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bearish streak exhaustion (>2) + volume spike + above 1d EMA50
            if (bear_streak_count[i] >= 2 and 
                volume_spike[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bullish streak exhaustion (>2) + volume spike + below 1d EMA50
            elif (bull_streak_count[i] >= 2 and 
                  volume_spike[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bullish streak resumes or price drops below 1d EMA50
            if (bull_streak_count[i] >= 1 or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bearish streak resumes or price rises above 1d EMA50
            if (bear_streak_count[i] >= 1 or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals