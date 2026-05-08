#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Donchian20_Trend_Volume_Spike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d = (close_1d > ema34_1d).astype(float)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Daily Donchian channels (20-period) for breakout levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period high and low for Donchian channels
    high_max = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    donchian_high = align_htf_to_ltf(prices, df_1d, high_max)
    donchian_low = align_htf_to_ltf(prices, df_1d, low_min)
    
    # Volume spike detection: current volume > 2.0 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above Donchian high with volume spike and 1d uptrend
            long_cond = (close[i] > donchian_high[i] and vol_spike[i] and trend_1d_aligned[i] > 0.5)
            
            # Short entry: price breaks below Donchian low with volume spike and 1d downtrend
            short_cond = (close[i] < donchian_low[i] and vol_spike[i] and trend_1d_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low (reversal signal)
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high (reversal signal)
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Donchian(20) breakout strategy on daily timeframe with volume spike confirmation and 1d EMA34 trend filter, executed on 12h timeframe.
# Enters long when price breaks above 20-day high with volume spike and 1d uptrend (close > EMA34).
# Enters short when price breaks below 20-day low with volume spike and 1d downtrend (close < EMA34).
# Exits when price reverses back through the opposite Donchian level.
# Uses 20-period volume MA with 2.0x threshold for volume confirmation to avoid overtrading.
# Targets 15-25 trades/year on 12h timeframe to minimize fee drag. Works in both bull markets (trend-following breakouts) 
# and bear markets (reversal breakouts from extreme levels). Uses discrete sizing (0.25) to minimize churn.