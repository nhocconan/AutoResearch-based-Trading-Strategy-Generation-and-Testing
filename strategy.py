#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d trend filter and volume spike confirmation.
# Donchian breakouts capture strong momentum moves, filtered by 1d EMA trend direction
# and volume confirmation to avoid false breakouts. Designed for ~25-35 trades/year
# per symbol, using 1h timeframe for entry timing with 4h/1h for signal direction.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Donchian channels (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # 1d EMA trend filter (50-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Apply session filter
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        # Long entry: price breaks above 4h Donchian high + above 1d EMA50 + volume spike
        if (close[i] > donchian_high_aligned[i] and 
            close[i] > ema50_1d_aligned[i] and 
            volume_filter[i]):
            if position != 1:
                signals[i] = 0.20
                position = 1
            else:
                signals[i] = 0.20
        
        # Short entry: price breaks below 4h Donchian low + below 1d EMA50 + volume spike
        elif (close[i] < donchian_low_aligned[i] and 
              close[i] < ema50_1d_aligned[i] and 
              volume_filter[i]):
            if position != -1:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = -0.20
        
        # Exit conditions: price returns to middle of Donchian channel
        elif position == 1 and close[i] < (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2:
            signals[i] = 0.0
            position = 0
        
        # Hold current position
        else:
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_DonchianBreakout_1dEMA50_VolumeFilter_Session"
timeframe = "1h"
leverage = 1.0