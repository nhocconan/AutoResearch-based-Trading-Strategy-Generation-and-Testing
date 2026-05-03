#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above 20-period Donchian high in 1d uptrend with volume spike (>1.5x 20-period volume MA).
# Short when price breaks below 20-period Donchian low in 1d downtrend with volume spike.
# Uses 1d EMA50 for higher timeframe trend alignment to avoid counter-trend trades.
# Volume spike confirms institutional participation. Designed for 4h timeframe to achieve 75-200 total trades over 4 years.
# Donchian channels provide clear structure-based breakouts effective in both trending and ranging markets.

name = "4h_Donchian20_1dEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on primary timeframe
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detection (20-period volume MA on primary timeframe)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)  # Volume at least 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any value is NaN
        if (np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        donchian_high = high_ma[i]
        donchian_low = low_ma[i]
        trend_up = close_val > ema_50_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_50_1d_aligned[i]  # 1d downtrend
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND 1d uptrend AND volume spike
            if close_val > donchian_high and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND 1d downtrend AND volume spike
            elif close_val < donchian_low and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            # Exit: price closes below Donchian low (reversal signal)
            if close_val < donchian_low:
                exit_signal = True
            # Exit: 1d trend changes to downtrend
            elif not trend_up:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions for short
            exit_signal = False
            # Exit: price closes above Donchian high (reversal signal)
            if close_val > donchian_high:
                exit_signal = True
            # Exit: 1d trend changes to uptrend
            elif not trend_down:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals