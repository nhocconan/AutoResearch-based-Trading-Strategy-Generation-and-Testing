#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation.
Long when price breaks above upper Donchian with 1d EMA34 uptrend and volume > 2.0x 20-period average.
Short when price breaks below lower Donchian with 1d EMA34 downtrend and volume > 2.0x 20-period average.
Exit on opposite Donchian band touch or trend reversal.
Uses discrete sizing (0.25) to minimize fee churn. Target: 20-50 trades/year on 4h.
Works in bull via trend-following breakouts, in bear via volatility expansion capture.
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
    
    # Get 4h data for Donchian calculations (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian channels for each 4h bar (based on previous 20 bars)
    upper_4h = np.full(len(close_4h), np.nan)
    lower_4h = np.full(len(close_4h), np.nan)
    
    for i in range(20, len(close_4h)):
        # Upper: highest high of previous 20 bars
        # Lower: lowest low of previous 20 bars
        upper_4h[i] = np.max(high_4h[i-20:i])
        lower_4h[i] = np.min(low_4h[i-20:i])
    
    # Align Donchian levels to original timeframe
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA34 for trend
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian with uptrend and volume spike
            long_signal = (close[i] > upper_4h_aligned[i]) and (close[i] > ema_34_1d_aligned[i]) and vol_spike[i]
            # Short: price breaks below lower Donchian with downtrend and volume spike
            short_signal = (close[i] < lower_4h_aligned[i]) and (close[i] < ema_34_1d_aligned[i]) and vol_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions: price touches lower band or trend reverses
            exit_signal = (close[i] < lower_4h_aligned[i]) or (close[i] < ema_34_1d_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: price touches upper band or trend reverses
            exit_signal = (close[i] > upper_4h_aligned[i]) or (close[i] > ema_34_1d_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0