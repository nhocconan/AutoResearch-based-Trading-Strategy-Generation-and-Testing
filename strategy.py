#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_volume_v2
# Strategy: 4h Camarilla pivot breakout with daily volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla levels act as institutional support/resistance; breakouts with volume capture institutional flow. Works in bull/bear as levels adapt to volatility. Target 20-40 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Using previous day's OHLC to avoid look-ahead
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels (based on previous day)
    # H5 = Close + 1.1 * (High - Low) * 1.1/2
    # H4 = Close + 1.1 * (High - Low) * 1.1
    # H3 = Close + 1.1 * (High - Low) * 1.1/0.5
    # L3 = Close - 1.1 * (High - Low) * 1.1/0.5
    # L4 = Close - 1.1 * (High - Low) * 1.1
    # L5 = Close - 1.1 * (High - Low) * 1.1/2
    
    # Calculate for previous day
    high_low_diff = prev_high - prev_low
    H3 = prev_close + 1.1 * high_low_diff * 1.1 / 0.5
    H4 = prev_close + 1.1 * high_low_diff * 1.1
    L3 = prev_close - 1.1 * high_low_diff * 1.1 / 0.5
    L4 = prev_close - 1.1 * high_low_diff * 1.1
    
    # Align to 4h timeframe (these levels are valid for the entire day)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # 4h Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(H3_aligned[i]) or np.isnan(H4_aligned[i]) or 
            np.isnan(L3_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout above H4 or below L3
        breakout_up = close[i] > H4_aligned[i]
        breakout_down = close[i] < L3_aligned[i]
        
        # Entry logic: Camarilla breakout with volume confirmation
        if breakout_up and vol_confirm[i] and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_down and vol_confirm[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite Camarilla level touch (mean reversion tendency)
        elif position == 1 and close[i] < L4_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > H3_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals