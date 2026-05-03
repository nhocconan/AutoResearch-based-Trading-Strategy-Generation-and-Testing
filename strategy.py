#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation.
# Long when price breaks above 1d Donchian upper band in 1w uptrend (price > EMA50) with volume spike.
# Short when price breaks below 1d Donchian lower band in 1w downtrend (price < EMA50) with volume spike.
# Uses discrete sizing 0.25 to balance return and drawdown. Target: 30-100 total trades over 4 years.
# Donchian channels provide strong trend-following structure, 1w EMA50 ensures higher timeframe alignment,
# Volume spike confirms institutional interest. Works in both bull and bear markets by only trading with
# the 1w trend, avoiding counter-trend whipsaws. Designed for 1d timeframe to minimize fee drag.

name = "1d_Donchian20_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper and lower bands
    donchian_upper_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (no alignment needed as same timeframe)
    donchian_upper_aligned = donchian_upper_1d
    donchian_lower_aligned = donchian_lower_1d
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike detection (20-period volume MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        vol_spike = volume_spike[i]
        trend_up = close_val > ema_50_1w_aligned[i]   # 1w uptrend
        trend_down = close_val < ema_50_1w_aligned[i]  # 1w downtrend
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND 1w uptrend AND volume spike
            if close_val > donchian_upper_aligned[i] and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band AND 1w downtrend AND volume spike
            elif close_val < donchian_lower_aligned[i] and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian lower band (reversal signal)
            if close_val < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian upper band (reversal signal)
            if close_val > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals