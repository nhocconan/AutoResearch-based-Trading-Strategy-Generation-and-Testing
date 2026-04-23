#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
- Long when price breaks above 20-day high AND close > 1w EMA34 AND volume > 1.5x 20-day average volume
- Short when price breaks below 20-day low AND close < 1w EMA34 AND volume > 1.5x 20-day average volume
- Exit when price crosses 10-day EMA (mean reversion to intermediate trend)
- Uses 1w EMA34 for HTF trend alignment to avoid counter-trend entries in bear markets
- Designed for low trade frequency (target: 7-25 trades/year) to minimize fee drag on 1d timeframe
- Volume confirmation reduces false breakouts; trend filter works in both bull and bear regimes
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
    
    # Calculate Donchian channels from 1d data (20-period)
    # We need to calculate rolling high/low on the 1d timeframe, then align
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 20-day high and low for Donchian channels
    donchian_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (already aligned since we're using 1d data)
    # But we need to align to the LTF (which is also 1d in this case, so it's direct)
    donchian_high_aligned = donchian_high  # Already at 1d frequency
    donchian_low_aligned = donchian_low    # Already at 1d frequency
    
    # Calculate 10-day EMA for exit signal
    ema10_1d = pd.Series(df_1d['close']).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema10_1d_aligned = ema10_1d  # Already at 1d frequency
    
    # Volume confirmation: > 1.5x 20-day average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need 34 for 1w EMA, 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema10_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high_aligned[i]  # Break above 20-day high
        breakout_down = close[i] < donchian_low_aligned[i]  # Break below 20-day low
        
        # Trend filter (using 1w EMA34)
        uptrend = close[i] > ema34_1w_aligned[i]
        downtrend = close[i] < ema34_1w_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 1.5 * vol_ma_20[i]
        
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
            # Exit: Price crosses 10-day EMA (mean reversion to intermediate trend)
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below 10-day EMA
                if close[i] < ema10_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: Price crosses above 10-day EMA
                if close[i] > ema10_1d_aligned[i]:
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