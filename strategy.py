#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wEMA34_Trend_VolumeSpike
Hypothesis: Daily Donchian(20) breakouts with 1-week EMA34 trend filter and volume spike confirmation. Uses discrete sizing (0.30) to limit trades (~15/year) and avoid fee drag. The 1-week EMA34 provides robust trend alignment across bull/bear markets. Volume spike (>2.0x 20-bar avg) confirms breakout momentum. Designed for BTC/ETH robustness via trend-following structure with strict entry conditions.
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
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA34 on 1w close for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate Donchian channels (20-day) on daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate 20-day average volume for confirmation on 1d
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian and volume MA20
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
        
        if position == 0:
            # Volume confirmation: current volume > 2.0x 20-day average (strict filter)
            volume_confirm = volume[i] > 2.0 * vol_ma20[i]
            
            # Long: price breaks above Donchian high in uptrend with volume spike
            # Short: price breaks below Donchian low in downtrend with volume spike
            long_signal = (close[i] > donchian_high_aligned[i]) and (close[i] > ema34_1w_aligned[i]) and volume_confirm
            short_signal = (close[i] < donchian_low_aligned[i]) and (close[i] < ema34_1w_aligned[i]) and volume_confirm
            
            if long_signal:
                signals[i] = 0.30
                position = 1
            elif short_signal:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.30
            # Exit when price moves back below 1w EMA34 (trend reversal)
            exit_signal = close[i] < ema34_1w_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            # Exit when price moves back above 1w EMA34 (trend reversal)
            exit_signal = close[i] > ema34_1w_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0