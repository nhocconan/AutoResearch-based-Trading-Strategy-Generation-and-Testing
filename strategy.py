#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian(20) breakout with 1w Supertrend trend filter and volume confirmation
# Long when price breaks above 1d Donchian upper(20) AND 1w Supertrend is bullish AND volume > 1.3 * avg_volume(20) on 4h
# Short when price breaks below 1d Donchian lower(20) AND 1w Supertrend is bearish AND volume > 1.3 * avg_volume(20) on 4h
# Exit when price crosses the 1d Donchian midpoint (upper+lower)/2
# Uses discrete sizing 0.25 to balance return and risk
# Target: 100-200 total trades over 4 years (25-50/year) for 4h timeframe
# 1d Donchian provides strong structural breakout levels that reduce whipsaw
# 1w Supertrend ensures we trade with the dominant weekly trend (works in bull/bear)
# Volume confirmation (1.3x) validates breakout strength while limiting overtrading

name = "4h_1dDonchian20_1wSupertrend_VolumeConfirm"
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
    
    # Get 1d data ONCE before loop for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need at least 20 completed 1d bars for Donchian
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid_20 = (donchian_high_20 + donchian_low_20) / 2.0
    
    # Align 1d Donchian to 4h timeframe (wait for completed 1d bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid_20)
    
    # Get 1w data ONCE before loop for Supertrend trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:  # Need at least 10 completed weekly bars for ATR
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Supertrend (ATR=10, mult=3.0)
    tr1 = pd.Series(high_1w - low_1w)
    tr2 = pd.Series(np.abs(high_1w - pd.Series(close_1w).shift(1)))
    tr3 = pd.Series(np.abs(low_1w - pd.Series(close_1w).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_10 = tr.ewm(alpha=1/10, adjust=False, min_periods=10).mean().values
    
    hl2 = (high_1w + low_1w) / 2.0
    upper_band = hl2 + (3.0 * atr_10)
    lower_band = hl2 - (3.0 * atr_10)
    
    supertrend = np.zeros_like(close_1w)
    supertrend_direction = np.ones_like(close_1w)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_1w)):
        if supertrend_direction[i-1] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
            if close_1w[i] < supertrend[i]:
                supertrend_direction[i] = -1
            else:
                supertrend_direction[i] = 1
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
            if close_1w[i] > supertrend[i]:
                supertrend_direction[i] = 1
            else:
                supertrend_direction[i] = -1
    
    # Supertrend is bullish when direction is 1 (price above supertrend line)
    supertrend_bullish = (supertrend_direction == 1)
    supertrend_aligned = align_htf_to_ltf(prices, df_1w, supertrend_bullish)
    
    # Calculate volume confirmation: volume > 1.3 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(supertrend_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Donchian high, 1w Supertrend bullish, volume confirmation, in session
            if (close[i] > donchian_high_aligned[i] and 
                supertrend_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian low, 1w Supertrend bearish, volume confirmation, in session
            elif (close[i] < donchian_low_aligned[i] and 
                  not supertrend_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 1d Donchian midpoint
            if close[i] < donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above 1d Donchian midpoint
            if close[i] > donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals