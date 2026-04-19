#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout + 1w EMA trend + volume confirmation
# Donchian breakout (20) captures trend continuation on daily timeframe
# 1w EMA (34) provides higher timeframe trend filter to avoid counter-trend trades
# 1d volume spike (>1.5x 20-day average) adds conviction to breakouts
# Exit on opposite Donchian band touch
# Designed for low-frequency, high-conviction trades to minimize fee drag
# Target: 10-25 trades/year to stay within optimal range for 1d timeframe
name = "1d_Donchian_1wEMA_Volume_v3"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1d Donchian channels (20-period)
    upper_donch = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_donch = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # 1d volume average (20-period)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # 1w EMA trend filter (34-period)
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 1d timeframe
    upper_donch_aligned = align_htf_to_ltf(prices, df_1d, upper_donch)
    lower_donch_aligned = align_htf_to_ltf(prices, df_1d, lower_donch)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1d, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(upper_donch_aligned[i]) or np.isnan(lower_donch_aligned[i]) or \
           np.isnan(vol_ma_1d_aligned[i]) or np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 1.5x 20-day average
        volume_filter = volume[i] > 1.5 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above upper Donchian + above 1w EMA + volume
            if close[i] > upper_donch_aligned[i] and close[i] > ema_1w_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower Donchian + below 1w EMA + volume
            elif close[i] < lower_donch_aligned[i] and close[i] < ema_1w_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price touches or goes below lower Donchian band
            if close[i] <= lower_donch_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches or goes above upper Donchian band
            if close[i] >= upper_donch_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals