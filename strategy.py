#!/usr/bin/env python3
# 4h_12h_camarilla_breakout_v1
# Strategy: 4h Camarilla breakout with 12h trend filter and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels (L3/L3/H3/H4) act as key support/resistance.
# Breakouts above H3 or below L3 with 12h trend alignment and volume surge capture
# institutional moves. Works in bull via breakouts above H3/H4, in bear via breaks below L3/L4.
# Volume confirmation ensures breakout authenticity. Low trade frequency (~20-40/year) minimizes fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_breakout_v1"
timeframe = "4h"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Load 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels from previous 1d OHLC
    # Camarilla: based on previous day's range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    # H4 = close + 1.5 * range * 1.1
    # H3 = close + 1.25 * range * 1.1
    # L3 = close - 1.25 * range * 1.1
    # L4 = close - 1.5 * range * 1.1
    H4 = close_1d + 1.5 * range_1d * 1.1
    H3 = close_1d + 1.25 * range_1d * 1.1
    L3 = close_1d - 1.25 * range_1d * 1.1
    L4 = close_1d - 1.5 * range_1d * 1.1
    
    # Align Camarilla levels to 4h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Volume confirmation: 4h volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_avg_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(ema_50[i]) or np.isnan(ema_50_12h_aligned[i]) or \
           np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or \
           np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend alignment: 4h close > EMA50 and 12h EMA50 > previous 12h EMA50 (rising)
        # For simplicity: 4h close > 4h EMA50 and 12h EMA50 sloping up
        ema_50_12h_prev = ema_50_12h_aligned[i-1] if i > 0 else ema_50_12h_aligned[i]
        trend_up = (close[i] > ema_50[i]) and (ema_50_12h_aligned[i] > ema_50_12h_prev)
        trend_down = (close[i] < ema_50[i]) and (ema_50_12h_aligned[i] < ema_50_12h_prev)
        
        # Breakout conditions
        breakout_up = (high[i] > H3_aligned[i]) or (high[i] > H4_aligned[i])
        breakout_down = (low[i] < L3_aligned[i]) or (low[i] < L4_aligned[i])
        
        # Entry conditions
        # Long: Breakout above H3/H4 with up-trend and volume confirmation
        if breakout_up and trend_up and vol_confirm[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Breakdown below L3/L4 with down-trend and volume confirmation
        elif breakout_down and trend_down and vol_confirm[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Trend reversal (close crosses EMA50 in opposite direction)
        elif position == 1 and close[i] < ema_50[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > ema_50[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals