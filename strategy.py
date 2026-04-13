#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h 12h Donchian breakout with 1d volume confirmation and 1w trend filter.
# Long when price breaks above 6h Donchian upper band (20) + 1d volume > 1.5x average + 1w close > 1w EMA50.
# Short when price breaks below 6h Donchian lower band (20) + 1d volume > 1.5x average + 1w close < 1w EMA50.
# Uses price breakouts with volume confirmation and higher timeframe trend filter to avoid false breakouts.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Donchian channels (20-period)
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    for i in range(20, n):
        donchian_upper[i] = np.max(high[i-20:i])
        donchian_lower[i] = np.min(low[i-20:i])
    
    # 1d data for volume and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    close_1d = df_1d['close'].values
    
    # 1d average volume (10-period)
    avg_volume_1d = np.full(len(volume_1d), np.nan)
    for i in range(10, len(volume_1d)):
        avg_volume_1d[i] = np.mean(volume_1d[i-10:i])
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 6h timeframe
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # Align 1w indicators to 6h timeframe
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(volume_1d_aligned[i]) or np.isnan(avg_volume_1d_aligned[i]) or
            np.isnan(close_1w_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_1d = volume_1d_aligned[i]
        avg_vol_1d = avg_volume_1d_aligned[i]
        close_1w = close_1w_aligned[i]
        ema_1w = ema_50_1w_aligned[i]
        
        # Volume confirmation: 1d volume > 1.5x average volume
        volume_confirm = vol_1d > 1.5 * avg_vol_1d
        
        # Trend filter: 1w close vs 1w EMA50
        uptrend = close_1w > ema_1w
        downtrend = close_1w < ema_1w
        
        if position == 0:
            # Long: breakout above upper band + uptrend + volume confirmation
            if (price > donchian_upper[i] and 
                uptrend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: breakdown below lower band + downtrend + volume confirmation
            elif (price < donchian_lower[i] and 
                  downtrend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: breakdown below lower band
            if price < donchian_lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: breakout above upper band
            if price > donchian_upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_1w_Donchian_Breakout_Volume_Trend"
timeframe = "6h"
leverage = 1.0