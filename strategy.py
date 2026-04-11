#!/usr/bin/env python3
# 12h_1w_camarilla_volume_v1
# Strategy: 12h Camarilla pivot breakout with weekly trend filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Camarilla levels from 1d act as intraday support/resistance. 
# Breakouts above H3 or below L3 with 1w trend alignment (price > EMA20 weekly) and 
# volume > 1.5x 20-period average capture institutional moves. 
# Weekly trend filter reduces false breakouts in choppy markets. 
# Designed for low trade frequency (~15-30/year) to minimize fee drift.
# Works in bull markets via long breakouts and bear markets via short breakdowns.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_camarilla_volume_v1"
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
    
    # Load 1d data for Camarilla calculation (OHLC from previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # H4 = C + 1.5 * (H-L), H3 = C + 1.0 * (H-L), L3 = C - 1.0 * (H-L), L4 = C - 1.5 * (H-L)
    # where C, H, L are from previous 1d bar
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels
    H3_1d = prev_close_1d + 1.0 * (prev_high_1d - prev_low_1d)
    L3_1d = prev_close_1d - 1.0 * (prev_high_1d - prev_low_1d)
    
    # Align Camarilla levels to 12h timeframe
    H3_1d_aligned = align_htf_to_ltf(prices, df_1d, H3_1d)
    L3_1d_aligned = align_htf_to_ltf(prices, df_1d, L3_1d)
    
    # Weekly EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_avg_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if np.isnan(H3_1d_aligned[i]) or np.isnan(L3_1d_aligned[i]) or \
           np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_avg_20[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions
        breakout_up = close[i] > H3_1d_aligned[i]
        breakout_down = close[i] < L3_1d_aligned[i]
        
        # Trend filter: price above/below weekly EMA20
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Entry conditions
        # Long: breakout above H3 AND uptrend AND volume confirmation
        if breakout_up and uptrend and vol_confirm[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: breakdown below L3 AND downtrend AND volume confirmation
        elif breakout_down and downtrend and vol_confirm[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite breakout (reversion to mean)
        elif position == 1 and breakout_down:
            position = 0
            signals[i] = 0.0
        elif position == -1 and breakout_up:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals