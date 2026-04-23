#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation.
- Long: Close > Upper Donchian(20) AND price > 1d EMA34 AND volume > 2.0x 24-period avg
- Short: Close < Lower Donchian(20) AND price < 1d EMA34 AND volume > 2.0x 24-period avg
- Exit: Opposite Donchian breakout OR price crosses 1d EMA34
- Uses 1d HTF for EMA34 and Donchian levels (calculated from prior completed 1d bar)
- Designed for low trade frequency (19-50/year) to minimize fee drag on 4h timeframe
- Works in bull (buy breakouts above upper band) and bear (sell breakdowns below lower band)
- Volume confirmation reduces false breakouts
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
    
    # Volume confirmation: > 2.0x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Calculate 1d EMA34 for trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian(20) levels from prior 1d bar (HTF = 1d)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper band: highest high of last 20 completed 1d bars
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 completed 1d bars
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (use prior completed 1d bar)
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 24)  # Need 34 for EMA, 20 for Donchian, 24 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(upper_20_aligned[i]) or
            np.isnan(lower_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Donchian breakout signals (using current close vs prior levels)
        breakout_up = close[i] > upper_20_aligned[i-1]  # Close above prior upper band
        breakout_down = close[i] < lower_20_aligned[i-1]  # Close below prior lower band
        
        if position == 0:
            # Long: Donchian upper breakout up AND price > 1d EMA34 AND volume confirmation
            if breakout_up and volume_confirm and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Donchian lower breakout down AND price < 1d EMA34 AND volume confirmation
            elif breakout_down and volume_confirm and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Donchian lower breakout down OR price < 1d EMA34 (trend flip)
            if breakout_down or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Donchian upper breakout up OR price > 1d EMA34 (trend flip)
            if breakout_up or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0