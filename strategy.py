#!/usr/bin/env python3
# 4h_camarilla_volume_breakout_v1
# Hypothesis: 4-hour Camarilla pivot breakouts with volume confirmation and daily trend filter work in both bull and bear markets.
# Uses 1-day Camarilla levels (H4/L4) for breakout entries, requiring volume > 2x 20-period average and 1-day EMA(50) trend alignment.
# Exits on reverse signal or when price returns to Pivot point. Position size 0.25.
# Target: 20-40 trades/year (80-160 over 4 years) with controlled risk.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_volume_breakout_v1"
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
    
    # 1-day EMA(50) for trend filter (calculated on daily closes)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma_20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Calculate 1-day Camarilla levels for current day
        # Need previous day's OHLC for today's Camarilla levels
        day_idx = i // 6  # 6 four-hour bars per day
        if day_idx < 1:
            continue
            
        # Get previous day's OHLC from 1d data
        prev_day_idx = day_idx - 1
        if prev_day_idx >= len(df_1d):
            continue
            
        prev_high = df_1d['high'].iloc[prev_day_idx]
        prev_low = df_1d['low'].iloc[prev_day_idx]
        prev_close = df_1d['close'].iloc[prev_day_idx]
        
        # Calculate Camarilla levels
        range_ = prev_high - prev_low
        if range_ <= 0:
            continue
            
        # Camarilla levels: H4/L4 are the key breakout levels
        h4 = prev_close + range_ * 1.1 / 2
        l4 = prev_close - range_ * 1.1 / 2
        pivot = (prev_high + prev_low + prev_close) / 3
        
        if position == 1:  # Long position
            # Exit: price returns to pivot or reverse signal
            if close[i] <= pivot or (close[i] < l4 and ema_50_1d_aligned[i] > close[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to pivot or reverse signal
            if close[i] >= pivot or (close[i] > h4 and ema_50_1d_aligned[i] < close[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above H4 with volume confirmation and uptrend
            if close[i] > h4 and volume[i] > vol_threshold[i] and ema_50_1d_aligned[i] < close[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below L4 with volume confirmation and downtrend
            elif close[i] < l4 and volume[i] > vol_threshold[i] and ema_50_1d_aligned[i] > close[i]:
                position = -1
                signals[i] = -0.25
    
    return signals