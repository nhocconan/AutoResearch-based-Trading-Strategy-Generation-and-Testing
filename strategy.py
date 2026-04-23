#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
Uses 1w EMA50 to determine trend direction (long only when price > EMA50, short only when price < EMA50).
Enter long when price breaks above Donchian(20) upper band in uptrend with volume confirmation.
Enter short when price breaks below Donchian(20) lower band in downtrend with volume confirmation.
Exit on opposite Donchian(10) breakout or trend reversal.
Designed for 1d timeframe to maintain 7-25 trades/year with low fee drag.
Uses discrete position sizing (0.30) to balance return and risk.
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
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate Donchian channels on 1d
    donchian_len = 20
    exit_len = 10
    
    # Upper band: highest high over last donchian_len periods
    highest_high = pd.Series(high).rolling(window=donchian_len, min_periods=donchian_len).max().values
    # Lower band: lowest low over last donchian_len periods
    lowest_low = pd.Series(low).rolling(window=donchian_len, min_periods=donchian_len).min().values
    # Exit bands (for opposite breakout exit)
    highest_high_exit = pd.Series(high).rolling(window=exit_len, min_periods=exit_len).max().values
    lowest_low_exit = pd.Series(low).rolling(window=exit_len, min_periods=exit_len).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, donchian_len, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma_20[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(highest_high_exit[i]) or np.isnan(lowest_low_exit[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND uptrend (price > 1w EMA50) AND volume spike
            if close[i] > highest_high[i] and close[i] > ema_50_aligned[i] and volume_spike[i]:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Donchian lower band AND downtrend (price < 1w EMA50) AND volume spike
            elif close[i] < lowest_low[i] and close[i] < ema_50_aligned[i] and volume_spike[i]:
                signals[i] = -0.30
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            if position == 1:
                # Exit long: price breaks below Donchian exit lower band OR trend reversal (price < 1w EMA50)
                if close[i] < lowest_low_exit[i] or close[i] < ema_50_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price breaks above Donchian exit upper band OR trend reversal (price > 1w EMA50)
                if close[i] > highest_high_exit[i] or close[i] > ema_50_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "1D_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0