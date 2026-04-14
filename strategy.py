#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and 1w trend filter
# Donchian(20) breakout provides clear entry/exit signals
# 1d volume surge (>1.5x 20-period average) confirms breakout strength
# 1w EMA(50) filter ensures trades align with higher timeframe trend
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue)
# Low trade frequency expected: ~20-40 trades/year per symbol
# Position size: 0.25 (25%) to balance return and drawdown

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h Donchian Channels (20 periods)
    dc_len = 20
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Upper band: highest high over past 20 periods
    upper = pd.Series(high_4h).rolling(window=dc_len, min_periods=dc_len).max().values
    # Lower band: lowest low over past 20 periods
    lower = pd.Series(low_4h).rolling(window=dc_len, min_periods=dc_len).min().values
    
    # Align Donchian channels to 4h timeframe (already at 4h, but keep for consistency)
    upper_aligned = upper  # Already aligned to 4h bars
    lower_aligned = lower  # Already aligned to 4h bars
    
    # Load 1d data ONCE for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d volume average (20 periods)
    vol_1d = df_1d['volume'].values
    vol_ma = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align volume MA to 4h timeframe
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    # Load 1w data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA (50 periods)
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, dc_len)  # Need at least 50 for 1w EMA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_surge = vol_current > (1.5 * vol_ma_aligned[i])
        
        # Trend filter: price above/below 1w EMA50
        above_ema = price > ema_50_aligned[i]
        below_ema = price < ema_50_aligned[i]
        
        if position == 0:
            # Enter long: price breaks above upper Donchian + volume surge + above weekly EMA
            if price > upper_aligned[i] and vol_surge and above_ema:
                position = 1
                signals[i] = position_size
            # Enter short: price breaks below lower Donchian + volume surge + below weekly EMA
            elif price < lower_aligned[i] and vol_surge and below_ema:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 50% of Donchian range OR trend changes
            donchian_mid = (upper_aligned[i] + lower_aligned[i]) / 2
            if price < donchian_mid or not above_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above 50% of Donchian range OR trend changes
            donchian_mid = (upper_aligned[i] + lower_aligned[i]) / 2
            if price > donchian_mid or not below_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_Volume_1wEMA_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0