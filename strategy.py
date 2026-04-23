#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
- Long when price breaks above 20-period Donchian high AND close > 1w EMA34 AND volume > 1.5x 20-period average
- Short when price breaks below 20-period Donchian low AND close < 1w EMA34 AND volume > 1.5x 20-period average
- Exit when price crosses 10-period EMA (mean reversion to intermediate trend)
- Uses 1w EMA34 for HTF trend alignment to avoid counter-trend entries in bear markets
- Volume confirmation reduces false breakouts
- Designed for low trade frequency (target: 12-37 trades/year) to minimize fee drag
- Works in both bull and bear markets: trend filter prevents counter-trend entries
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
    
    # Get 1w data for EMA34 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1w EMA34
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate Donchian channels (20-period) on 12h timeframe
    # Highest high and lowest low over past 20 periods
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 10-period EMA for exit signal
    ema10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 20, 10)  # Need 34 for EMA34, 20 for Donchian/volume, 10 for EMA10
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(ema10[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high[i]  # Break above upper Donchian
        breakout_down = close[i] < donchian_low[i]  # Break below lower Donchian
        
        # Trend filter (using 1w EMA34)
        uptrend = close[i] > ema34_1w_aligned[i]
        downtrend = close[i] < ema34_1w_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Donchian breakout up + uptrend + volume confirmation
            if breakout_up and uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down + downtrend + volume confirmation
            elif breakout_down and downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses 10-period EMA (mean reversion)
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below EMA10
                if close[i] < ema10[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: Price crosses above EMA10
                if close[i] > ema10[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_1wEMA34_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0