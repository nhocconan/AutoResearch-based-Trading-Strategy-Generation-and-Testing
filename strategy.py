# State your hypothesis in a comment at the top (strategy type, timeframe, why it should work in BOTH bull AND bear)
# Hypothesis: 1-day price action with weekly trend filter. Long when price breaks above weekly Donchian high AND daily close > weekly EMA (uptrend). Short when price breaks below weekly Donchian low AND daily close < weekly EMA (downtrend). Exit when price crosses back inside weekly Donchian channel. Uses 1d timeframe as required, with 1w Donchian and EMA for trend context. Designed to capture multi-week trends while avoiding whipsaws in ranging markets. Target: 10-30 trades per year (40-120 total over 4 years) to minimize fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyDonchian_EMA_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Weekly data for Donchian channel and EMA
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 20:
        return np.zeros(n)
    
    # Weekly Donchian channel (20-period)
    donch_high = pd.Series(df_w['high'].values).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_w['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA (50-period) for trend filter
    ema_w = pd.Series(df_w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly indicators to daily timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_w, donch_low)
    ema_w_aligned = align_htf_to_ltf(prices, df_w, ema_w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for weekly EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above weekly Donchian high AND daily close > weekly EMA (uptrend)
            long_cond = (high[i] > donch_high_aligned[i]) and (close[i] > ema_w_aligned[i])
            # Short conditions: price breaks below weekly Donchian low AND daily close < weekly EMA (downtrend)
            short_cond = (low[i] < donch_low_aligned[i]) and (close[i] < ema_w_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below weekly Donchian low
            if low[i] < donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above weekly Donchian high
            if high[i] > donch_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals