#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
- Long when price breaks above Donchian upper band (20-day high) AND close > 1w EMA34 AND volume > 1.5x 20-day average
- Short when price breaks below Donchian lower band (20-day low) AND close < 1w EMA34 AND volume > 1.5x 20-day average
- Exit when price crosses Donchian middle band (20-day average of high/low)
- Uses 1w EMA34 for HTF trend alignment to avoid counter-trend entries in bear markets
- Volume confirmation reduces false breakouts
- Target: 7-25 trades/year (30-100 total over 4 years) to minimize fee drag
- Works in both bull and bear markets: trend filter ensures we only trade with the dominant weekly trend
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
    
    # Calculate Donchian channels from daily data (20-period)
    # We need to resample to daily equivalent using 1d data from mtf_data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Donchian upper band: 20-day high
    donchian_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    # Donchian lower band: 20-day low
    donchian_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    # Donchian middle band: average of upper and lower
    donchian_middle = (donchian_high + donchian_low) / 2
    
    # Align Donchian levels to 1d timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1d, donchian_middle)
    
    # Volume confirmation: > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need 34 for EMA34, 20 for Donchian/volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high_aligned[i]  # Break above upper band
        breakout_down = close[i] < donchian_low_aligned[i]  # Break below lower band
        
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
            # Exit: Price crosses Donchian middle band (mean reversion to center)
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below middle band
                if close[i] < donchian_middle_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: Price crosses above middle band
                if close[i] > donchian_middle_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_1wEMA34_Trend_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0