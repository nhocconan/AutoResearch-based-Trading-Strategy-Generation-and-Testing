#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above Donchian upper band AND close > 1w EMA50 AND volume > 2.0x 20-period average.
Short when price breaks below Donchian lower band AND close < 1w EMA50 AND volume > 2.0x 20-period average.
Exit when price crosses Donchian middle band (20-period SMA).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 20-50 trades/year per symbol.
Donchian channels provide proven breakout edge on BTC/ETH pairs across market regimes.
1w EMA50 offers smooth trend filter for 1d timeframe alignment with lower lag than monthly.
Volume confirmation at 2.0x ensures only institutional-grade breakouts are taken.
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
    
    # Load 1w data for EMA50 - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Donchian channels on 1d timeframe (20-period)
    # We need to compute these on 1d data then align to 1d primary timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian upper band (20-period high)
    upper_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Donchian lower band (20-period low)
    lower_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    # Donchian middle band (20-period SMA of close)
    middle_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align Donchian levels to 1d timeframe (primary)
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    middle_1d_aligned = align_htf_to_ltf(prices, df_1d, middle_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 20)  # Ensure warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_1d_aligned[i]) or np.isnan(lower_1d_aligned[i]) or 
            np.isnan(middle_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND close > 1w EMA50 AND volume spike
            if (price > upper_1d_aligned[i] and 
                close[i] > ema50_1w_aligned[i] and 
                volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band AND close < 1w EMA50 AND volume spike
            elif (price < lower_1d_aligned[i] and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume[i] > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: price crosses Donchian middle band
            if position == 1 and price < middle_1d_aligned[i]:
                exit_signal = True
            elif position == -1 and price > middle_1d_aligned[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0