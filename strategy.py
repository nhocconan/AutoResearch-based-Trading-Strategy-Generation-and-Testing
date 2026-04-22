#!/usr/bin/env python3
"""
Hypothesis: 1-hour Bollinger Band Squeeze Breakout with 4-hour Trend and Volume Confirmation.
Long when price breaks above upper Bollinger Band during low volatility (squeeze) and 4h EMA50 rising with volume spike.
Short when price breaks below lower Bollinger Band during low volatility and 4h EMA50 falling with volume spike.
Exit when price returns to Bollinger Band middle or 4h EMA50 reverses.
Designed for low trade frequency by requiring volatility contraction (squeeze) before breakout.
Works in both bull and bear markets by following the 4h trend direction.
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
    
    # Bollinger Bands (20, 2)
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Bollinger Band Width for squeeze detection (low volatility)
    bb_width = (bb_upper - bb_lower) / bb_middle
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < 0.8 * bb_width_ma  # Bollinger Band width below 80% of its 20-period average
    
    # Load 4h data for trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 50-period EMA on 4h close for trend
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_middle[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above upper BB during squeeze, 4h EMA50 rising, volume spike
            if (close[i] > bb_upper[i] and squeeze[i] and 
                ema50_4h_aligned[i] > ema50_4h_aligned[i-1] and vol_spike):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below lower BB during squeeze, 4h EMA50 falling, volume spike
            elif (close[i] < bb_lower[i] and squeeze[i] and 
                  ema50_4h_aligned[i] < ema50_4h_aligned[i-1] and vol_spike):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: price returns to BB middle or 4h EMA50 reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price <= BB middle or 4h EMA50 turns down
                if close[i] <= bb_middle[i] or ema50_4h_aligned[i] < ema50_4h_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price >= BB middle or 4h EMA50 turns up
                if close[i] >= bb_middle[i] or ema50_4h_aligned[i] > ema50_4h_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_BollingerSqueeze_4hTrend_Volume_Session"
timeframe = "1h"
leverage = 1.0