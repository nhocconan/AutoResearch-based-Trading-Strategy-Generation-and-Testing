#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and 1w trend filter
# Weekly trend (1w close > 1w SMA10) filters for momentum direction
# Daily volume > 1.5x average confirms institutional participation
# 4h Donchian(20) breakout captures momentum in trend direction
# Works in bull/bear as weekly trend adapts to market regime
# Target: 20-30 trades/year per symbol (80-120 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d volume MA (20-period)
    vol_ma = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    # Calculate 1w trend: close > SMA10
    sma10_1w = pd.Series(df_1w['close']).rolling(window=10, min_periods=10).mean().values
    trend_1w = df_1w['close'].values > sma10_1w
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w.astype(float))
    
    # 4h Donchian channel (20 periods)
    dc_len = 20
    dc_upper = pd.Series(high).rolling(window=dc_len, min_periods=dc_len).max().shift(1).values
    dc_lower = pd.Series(low).rolling(window=dc_len, min_periods=dc_len).min().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, dc_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(dc_upper[i]) or 
            np.isnan(dc_lower[i]) or
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(trend_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x average
        volume_confirmed = df_1d['volume'].iloc[i // 24] > 1.5 * vol_ma_aligned[i] if i // 24 < len(df_1d) else False
        
        if position == 0:
            # Enter long: Donchian breakout above + up trend + volume
            if (close[i] > dc_upper[i] and 
                trend_1w_aligned[i] > 0.5 and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: Donchian breakdown below + down trend + volume
            elif (close[i] < dc_lower[i] and 
                  trend_1w_aligned[i] < 0.5 and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian lower or trend changes
            if close[i] < dc_lower[i] or trend_1w_aligned[i] < 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Donchian upper or trend changes
            if close[i] > dc_upper[i] or trend_1w_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_1w_Donchian_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0