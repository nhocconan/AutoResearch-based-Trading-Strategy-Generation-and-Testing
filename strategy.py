#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above Donchian upper band AND close > 1w EMA50 AND volume > 2.0x 20-period average.
Short when price breaks below Donchian lower band AND close < 1w EMA50 AND volume > 2.0x 20-period average.
Exit when price crosses Donchian midpoint (mean of upper and lower bands).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 15-25 trades/year per symbol.
The 1w EMA50 provides a long-term trend filter that works in both bull and bear markets.
Volume confirmation at 2.0x ensures only high-momentum breakouts are taken, reducing false signals.
Donchian levels from 1d provide clear price structure with defined risk/reward.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Donchian levels - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period) on 1d data
    # Upper band = highest high over 20 periods
    # Lower band = lowest low over 20 periods
    upper_series = pd.Series(high_1d).rolling(window=20, min_periods=20).max()
    lower_series = pd.Series(low_1d).rolling(window=20, min_periods=20).min()
    upper_1d = upper_series.values
    lower_1d = lower_series.values
    
    # Align Donchian levels to 1d timeframe (no conversion needed, already 1d)
    upper_aligned = upper_1d  # Already aligned to 1d bars
    lower_aligned = lower_1d  # Already aligned to 1d bars
    
    # Calculate midpoint for exit
    midpoint_1d = (upper_1d + lower_1d) / 2
    midpoint_aligned = midpoint_1d  # Already aligned to 1d bars
    
    # Volume average (20-period) on 1d timeframe
    vol_ma_series = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_ma = vol_ma_series.values
    
    # Load 1w data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w data
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(midpoint_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND close > 1w EMA50 AND volume spike
            if (price > upper_aligned[i] and 
                close[i] > ema50_1w_aligned[i] and 
                volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band AND close < 1w EMA50 AND volume spike
            elif (price < lower_aligned[i] and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume[i] > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: price crosses Donchian midpoint
            if position == 1 and price < midpoint_aligned[i]:
                exit_signal = True
            elif position == -1 and price > midpoint_aligned[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_1wEMA50_VolumeConfirm"
timeframe = "1d"
leverage = 1.0