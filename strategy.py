#!/usr/bin/env python3
"""
1h_101174 - 1h timeframe with 4h and 1d filters
Hypothesis: Use 4h Donchian breakout for trend direction and 1d volume spike for confirmation,
with 1h RSI for precise entry timing. This avoids overtrading by using higher timeframes
for signal generation while using 1h only for entry timing refinement. Designed to work
in both bull and bear markets by requiring volume confirmation and trend alignment.
Target: 15-35 trades per year to minimize fee drag.
"""

name = "1h_101174"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h data for trend direction (Donchian breakout)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # 1d volume spike detection (current volume > 2x 20-day average)
    volume_1d = df_1d['volume'].values
    vol_ma20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (vol_ma20 * 2.0)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # 1h RSI for entry timing (14-period)
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Donchian high + volume spike + RSI not overbought + session
            if (close[i] > donchian_high_aligned[i] and 
                volume_spike_aligned[i] and 
                rsi[i] < 70 and 
                session_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian low + volume spike + RSI not oversold + session
            elif (close[i] < donchian_low_aligned[i] and 
                  volume_spike_aligned[i] and 
                  rsi[i] > 30 and 
                  session_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below 4h Donchian low or RSI overbought
            if close[i] < donchian_low_aligned[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above 4h Donchian high or RSI oversold
            if close[i] > donchian_high_aligned[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals