#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Donchian_Breakout_Trend_Volume
Hypothesis: Trade 6h Donchian(20) breakouts in the direction of weekly Camarilla pivot trend (price > R3 = uptrend, price < S3 = downtrend) with volume confirmation (1.5x average). Uses ATR trailing stop (2.0) for risk management. Designed for low trade frequency (~12-30/year) by requiring strong confluence: breakout + weekly trend + volume spike. Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend). Focus on BTC/ETH as primary targets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF filters
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly Camarilla levels from previous week
    prev_high_1w = df_1w['high'].shift(1).values
    prev_low_1w = df_1w['low'].shift(1).values
    prev_close_1w = df_1w['close'].shift(1).values
    
    camarilla_range_1w = prev_high_1w - prev_low_1w
    R3_1w = prev_close_1w + camarilla_range_1w * 3.0/12
    S3_1w = prev_close_1w - camarilla_range_1w * 3.0/12
    
    # Align weekly indicators to 6h timeframe
    R3_1w_aligned = align_htf_to_ltf(prices, df_1w, R3_1w)
    S3_1w_aligned = align_htf_to_ltf(prices, df_1w, S3_1w)
    
    # 6h Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop (14-period on 6h)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Warmup: max of weekly Camarilla (2), Donchian (20), volume MA (20), ATR (14)
    start_idx = max(2, 20, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R3_1w_aligned[i]) or 
            np.isnan(S3_1w_aligned[i]) or 
            np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr_14[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        R3_val = R3_1w_aligned[i]
        S3_val = S3_1w_aligned[i]
        high_20_val = high_20[i]
        low_20_val = low_20[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        atr_14_val = atr_14[i]
        
        if position == 0:
            # Long: break above Donchian high, weekly uptrend (close > R3), volume spike
            long_signal = (high_val > high_20_val) and (close_val > R3_val) and (volume_val > 1.5 * vol_ma_val)
            # Short: break below Donchian low, weekly downtrend (close < S3), volume spike
            short_signal = (low_val < low_20_val) and (close_val < S3_val) and (volume_val > 1.5 * vol_ma_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                long_stop = entry_price - 2.0 * atr_14_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                short_stop = entry_price + 2.0 * atr_14_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Update trailing stop: move stop up as price makes new highs
            long_stop = max(long_stop, high_val - 2.0 * atr_14_val)
            # Exit: trailing stop hit or trend reversal (close < R3)
            if (low_val < long_stop) or (close_val < R3_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, low_val + 2.0 * atr_14_val)
            # Exit: trailing stop hit or trend reversal (close > S3)
            if (high_val > short_stop) or (close_val > S3_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Weekly_Pivot_Donchian_Breakout_Trend_Volume"
timeframe = "6h"
leverage = 1.0