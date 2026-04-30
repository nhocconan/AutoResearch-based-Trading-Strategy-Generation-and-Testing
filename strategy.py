#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1w Supertrend trend filter and volume confirmation.
# Long when price breaks above Donchian upper (20), close > 1w Supertrend, and volume > 1.5x 20-bar avg.
# Short when price breaks below Donchian lower (20), close < 1w Supertrend, and volume > 1.5x 20-bar avg.
# Exit when price crosses the Donchian midpoint (mean of upper and lower).
# Uses 4h timeframe for optimal trade frequency (target: 20-50 trades/year) to minimize fee drag.
# Donchian channels provide clear breakout levels based on recent price extremes.
# 1w Supertrend filters for higher timeframe trend alignment (avoids counter-trend trades).
# Volume confirmation with moderate threshold reduces false breakouts while keeping trade count reasonable.
# Works in bull markets via breakouts with uptrend and in bear markets via breakdowns with downtrend.
# Target: 75-200 total trades over 4 years.

name = "4h_Donchian20_1wSupertrend_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for Supertrend trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w Supertrend for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # ATR calculation for Supertrend
    tr1 = pd.Series(high_1w - low_1w)
    tr2 = pd.Series(np.abs(high_1w - close_1w.shift(1)))
    tr3 = pd.Series(np.abs(low_1w - close_1w.shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=10, adjust=False, min_periods=10).mean()
    
    # Supertrend calculation
    upperband = (high_1w + low_1w) / 2 + 3.0 * atr
    lowerband = (high_1w + low_1w) / 2 - 3.0 * atr
    
    supertrend = np.zeros_like(close_1w)
    direction = np.ones_like(close_1w)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upperband[0]
    direction[0] = 1
    
    for i in range(1, len(close_1w)):
        if close_1w[i] > supertrend[i-1]:
            supertrend[i] = upperband[i]
            direction[i] = 1
        else:
            supertrend[i] = lowerband[i]
            direction[i] = -1
    
    # Align Supertrend to 4h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1w, supertrend)
    
    # Calculate Donchian channels (20-period) on 4h data
    donchian_window = 20
    donchian_upper = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_lower = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donchian_window, 20)  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(supertrend_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(donchian_middle[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_supertrend = supertrend_aligned[i]
        curr_upper = donchian_upper[i]
        curr_lower = donchian_lower[i]
        curr_middle = donchian_middle[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper, close > Supertrend (uptrend), volume spike
            if (curr_close > curr_upper and 
                curr_close > curr_supertrend and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower, close < Supertrend (downtrend), volume spike
            elif (curr_close < curr_lower and 
                  curr_close < curr_supertrend and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price crosses below Donchian midpoint
            if curr_close < curr_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price crosses above Donchian midpoint
            if curr_close > curr_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals