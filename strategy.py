#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume spike and EMA trend filter.
# Long when price breaks above 1d Donchian upper band AND 1w volume spike AND price > 1w EMA50.
# Short when price breaks below 1d Donchian lower band AND 1w volume spike AND price < 1w EMA50.
# Uses weekly volume for momentum confirmation and weekly EMA for trend direction.
# Designed for very few trades (target: 10-20/year) to minimize fee drag and improve generalization.
# Works in both bull and bear markets by following weekly trend with volatility filter.
name = "1d_Donchian20_1wVolume_EMA50"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for volume spike and EMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # 1w volume spike: current volume > 2.0 * 20-period EMA
    vol_ema_20 = pd.Series(df_1w['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike_1w = np.where(vol_ema_20 > 0, df_1w['volume'].values / vol_ema_20, 1.0) > 2.0
    vol_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_spike_1w)
    
    # 1w EMA50 for trend filter
    ema_50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # 1d Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Sufficient warmup for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(vol_spike_1w_aligned[i]) or np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long condition: break above upper band, volume spike, uptrend (price > EMA50)
            long_condition = (close[i] > highest_high_20[i]) and vol_spike_1w_aligned[i] and (close[i] > ema_50_aligned[i])
            # Short condition: break below lower band, volume spike, downtrend (price < EMA50)
            short_condition = (close[i] < lowest_low_20[i]) and vol_spike_1w_aligned[i] and (close[i] < ema_50_aligned[i])
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below lower band or trend reverses (price < EMA50)
            if (close[i] < lowest_low_20[i]) or (close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above upper band or trend reverses (price > EMA50)
            if (close[i] > highest_high_20[i]) or (close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals