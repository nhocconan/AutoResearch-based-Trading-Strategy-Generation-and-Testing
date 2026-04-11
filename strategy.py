#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_v1
# Strategy: 4h Camarilla pivot breakout with volume confirmation and 1d trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels (H3/L3) act as strong support/resistance.
# Breakouts above H3 or below L3 with volume confirmation signal momentum.
# 1d EMA50 trend filter ensures trades align with higher timeframe direction.
# Designed for low trade frequency (~20-50/year) to minimize fee drag.
# Works in bull markets via long breakouts above H3 in uptrend,
# and bear markets via short breakdowns below L3 in downtrend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h typical price for Camarilla calculation
    typical_price = (high + low + close) / 3.0
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels using previous day's OHLC
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # We use previous day's values to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    H3 = prev_close + 1.1 * (prev_high - prev_low)
    L3 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Align Camarilla levels to 4h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Volume confirmation: current 4h volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or \
           np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg_20[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Trend filter: 1d EMA50 direction
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > H3_aligned[i]  # Price breaks above H3
        breakdown_down = close[i] < L3_aligned[i]  # Price breaks below L3
        
        # Entry conditions
        # Long: Breakout above H3 AND uptrend AND volume confirmation
        if breakout_up and uptrend and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Breakdown below L3 AND downtrend AND volume confirmation
        elif breakdown_down and downtrend and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite breakout or trend reversal
        elif position == 1 and (close[i] < L3_aligned[i] or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > H3_aligned[i] or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals