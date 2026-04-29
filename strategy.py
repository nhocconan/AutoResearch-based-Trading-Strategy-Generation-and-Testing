#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume spike confirmation
# Donchian breakouts capture strong momentum moves; 12h EMA50 filters for higher-timeframe trend alignment
# Volume spike confirms institutional participation. Works in bull markets via breakouts and
# in bear markets via breakdowns. Target: 20-50 trades/year (80-200 total over 4 years).

name = "4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 12h calculations
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average (strict filter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for 12h EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_highest_high = highest_high[i]
        curr_lowest_low = lowest_low[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: price closes above upper Donchian + above 12h EMA50 + volume spike
            if (curr_close > curr_highest_high and 
                curr_close > curr_ema_50_12h and 
                curr_volume_spike):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price closes below lower Donchian + below 12h EMA50 + volume spike
            elif (curr_close < curr_lowest_low and 
                  curr_close < curr_ema_50_12h and 
                  curr_volume_spike):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position - exit on Donchian middle or trend change
            # Exit when price closes below Donchian middle OR below 12h EMA50
            donchian_middle = (curr_highest_high + curr_lowest_low) / 2.0
            if curr_close < donchian_middle or curr_close < curr_ema_50_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit on Donchian middle or trend change
            # Exit when price closes above Donchian middle OR above 12h EMA50
            donchian_middle = (curr_highest_high + curr_lowest_low) / 2.0
            if curr_close > donchian_middle or curr_close > curr_ema_50_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals