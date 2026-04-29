#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian breakout with 4h trend filter and 1d volume spike
# Donchian(20) breakout captures momentum; 4h EMA50 filters trend direction;
# 1d volume spike confirms institutional interest. Works in bull/bear by
# only taking breakouts in direction of higher timeframe trend.
# Target: 15-37 trades/year (60-150 total over 4 years).

name = "1h_Donchian20_Breakout_4hTrend_1dVolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d volume average for spike detection
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 1h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for 4h EMA, 1d volume MA, Donchian
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if HTF data not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_volume = volume[i]
        curr_ema_4h = ema_4h_aligned[i]
        curr_vol_ma_1d = vol_ma_1d_aligned[i]
        curr_highest_high = highest_high[i]
        curr_lowest_low = lowest_low[i]
        
        # Volume spike condition: current volume > 2.0x 1d average
        volume_spike = curr_volume > (2.0 * curr_vol_ma_1d)
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: price > 20-period high + 4h uptrend + volume spike
            if (curr_close > curr_highest_high and 
                curr_close > curr_ema_4h and 
                volume_spike):
                signals[i] = 0.20
                position = 1
            # Short breakout: price < 20-period low + 4h downtrend + volume spike
            elif (curr_close < curr_lowest_low and 
                  curr_close < curr_ema_4h and 
                  volume_spike):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price breaks below 20-period low OR 4h trend turns down
            if curr_close < curr_lowest_low or curr_close < curr_ema_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price breaks above 20-period high OR 4h trend turns up
            if curr_close > curr_highest_high or curr_close > curr_ema_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals