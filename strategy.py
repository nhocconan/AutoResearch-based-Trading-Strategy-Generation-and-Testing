#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend filter + volume spike confirmation
# Donchian breakout captures strong momentum moves; 1w EMA50 ensures alignment with weekly trend
# Volume spike (2.0x 20-period average) confirms institutional participation
# Discrete sizing 0.25 minimizes fee churn. Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_Donchian20_1wEMA50_VolumeSpike_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 1w EMA50 for trend filter (loaded ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 20  # warmup for Donchian channels
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_highest_20 = highest_20[i]
        curr_lowest_20 = lowest_20[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish breakout: price > highest_20 AND close > 1w EMA50 (uptrend)
                if curr_close > curr_highest_20 and curr_close > curr_ema_50_1w:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish breakout: price < lowest_20 AND close < 1w EMA50 (downtrend)
                elif curr_close < curr_lowest_20 and curr_close < curr_ema_50_1w:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price closes below lowest_20 (breakdown) OR closes below 1w EMA50
            if curr_close < curr_lowest_20 or curr_close < curr_ema_50_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price closes above highest_20 (breakout) OR closes above 1w EMA50
            if curr_close > curr_highest_20 or curr_close > curr_ema_50_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals