#!/usr/bin/env python3
# 4h_1d_camarilla_pullback_v1
# Strategy: 4h pullback to Camarilla support/resistance with 1d trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: In trending markets, price pulls back to Camarilla H3/L3 levels before resuming trend.
# Uses 1d EMA50 for trend direction and waits for pullback to H3/L3 with volume confirmation.
# Designed for low trade frequency (~20-30/year) by requiring trend alignment and pullback to specific levels.
# Works in both bull and bear markets by following the 1d trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_pullback_v1"
timeframe = "4h"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: H3, L3 (key support/resistance for pullbacks)
    rng = prev_high - prev_low
    H3 = prev_close + 1.1 * rng / 4
    L3 = prev_close - 1.1 * rng / 4
    
    # Align Camarilla levels to 4h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 20-period volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume[i] > 1.3 * vol_avg_20[i]
        
        # Pullback conditions
        # Pullback to H3 in uptrend: price touches H3 from below and bounces
        pullback_to_H3 = (low[i] <= H3_aligned[i]) and (close[i] > H3_aligned[i] * 0.999)
        # Pullback to L3 in downtrend: price touches L3 from above and bounces
        pullback_to_L3 = (high[i] >= L3_aligned[i]) and (close[i] < L3_aligned[i] * 1.001)
        
        # 1d EMA trend filter
        trend_bullish = close[i] > ema_50_1d_aligned[i]
        trend_bearish = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        # Long: Pullback to H3 in uptrend with volume confirmation
        if pullback_to_H3 and trend_bullish and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Pullback to L3 in downtrend with volume confirmation
        elif pullback_to_L3 and trend_bearish and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Price moves against trend beyond H4/L4 levels
        elif position == 1 and close[i] < ema_50_1d_aligned[i]:  # Below EMA = trend change
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > ema_50_1d_aligned[i]:  # Above EMA = trend change
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals