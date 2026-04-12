#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_12h_camarilla_volume_trend_v1
# Combines 12h Camarilla pivot levels with 4h volume confirmation and trend filter.
# Uses 12h trend direction (EMA50) to align with higher timeframe momentum.
# In bull markets: long when price > 12h EMA50 and breaks above H4.
# In bear markets: short when price < 12h EMA50 and breaks below L4.
# Volume confirmation ensures institutional participation.
# Target: 25-40 trades/year per symbol for low friction and high win rate.
name = "4h_12h_camarilla_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend and Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for Camarilla levels (standard practice)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    range_prev = high_prev - low_prev
    camarilla_h4 = close_prev + range_prev * 1.1 / 2
    camarilla_l4 = close_prev - range_prev * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    h4_level = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_level = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup for EMA50
        # Skip if levels not ready
        if np.isnan(h4_level[i]) or np.isnan(l4_level[i]) or np.isnan(ema_50_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Skip if volume confirmation fails
        if not vol_confirm[i]:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price > 12h EMA50 (bullish trend) and breaks above H4
        if close[i] > ema_50_12h_aligned[i] and close[i] > h4_level[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price < 12h EMA50 (bearish trend) and breaks below L4
        elif close[i] < ema_50_12h_aligned[i] and close[i] < l4_level[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: trend reversal or opposite breakout
        elif (close[i] < ema_50_12h_aligned[i] and position == 1) or \
             (close[i] > ema_50_12h_aligned[i] and position == -1):
            position = 0
            signals[i] = 0.0
        elif close[i] < l4_level[i] and position == 1:
            position = 0
            signals[i] = 0.0
        elif close[i] > h4_level[i] and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals