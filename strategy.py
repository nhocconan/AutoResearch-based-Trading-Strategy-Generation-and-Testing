#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
- Long when price breaks above 20-period Donchian high AND close > 1d EMA34 AND volume > 1.8x 20-period average
- Short when price breaks below 20-period Donchian low AND close < 1d EMA34 AND volume > 1.8x 20-period average
- Exit when price crosses 10-period EMA (mean reversion to short-term trend)
- Uses 1d EMA34 for HTF trend alignment to avoid counter-trend entries in bear markets
- Designed for both bull and bear markets: trend filter prevents counter-trend entries
- Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
- Works on BTC/ETH/SOL by focusing on structure and volume confirmation
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
    
    # Get 1d data for EMA34 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 10-period EMA for exit (LTf)
    close_series = pd.Series(close)
    ema10 = close_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate Donchian channels (20-period) from 12h data
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    donchian_high = pd.Series(df_12h['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_12h['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 20)  # Need 34 for EMA34, 20 for Donchian, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema10[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high_aligned[i]  # Break above Donchian high
        breakout_down = close[i] < donchian_low_aligned[i]  # Break below Donchian low
        
        # Trend filter (using 1d EMA34)
        uptrend = close[i] > ema34_1d_aligned[i]
        downtrend = close[i] < ema34_1d_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 1.8 * vol_ma[i]
        
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
                # Exit long: Price crosses below 10 EMA
                if close[i] < ema10[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: Price crosses above 10 EMA
                if close[i] > ema10[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_1dEMA34_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0