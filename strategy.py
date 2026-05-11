#!/usr/bin/env python3
name = "1h_Camarilla_R1S1_4hTrend_1dVolumeSpike"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla levels (calculated on previous day's close)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for trend and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels from previous day's OHLC
    # Using previous day's close, high, low to calculate today's levels
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]  # first value
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    # Camarilla R1, S1 levels
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Calculate 1d EMA34 for trend
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d volume spike (volume > 1.5x 20-period average)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (vol_ma_1d * 1.5)
    
    # Align all 1d data to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Calculate 4h close for trend filter
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    signals = np.zeros(n)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if not in_session[i]:
            if i > 0:
                signals[i] = signals[i-1]  # hold position outside session
            else:
                signals[i] = 0.0
            continue
            
        if np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or \
           np.isnan(ema34_1d_aligned[i]) or np.isnan(ema20_4h_aligned[i]):
            if i > 0:
                signals[i] = signals[i-1]
            else:
                signals[i] = 0.0
            continue
        
        # Long conditions: price > S1, above 1d EMA34, above 4h EMA20, volume spike
        if (close[i] > camarilla_s1_aligned[i] and 
            close[i] > ema34_1d_aligned[i] and 
            close[i] > ema20_4h_aligned[i] and 
            vol_spike_1d_aligned[i]):
            signals[i] = 0.20
        # Short conditions: price < R1, below 1d EMA34, below 4h EMA20, volume spike
        elif (close[i] < camarilla_r1_aligned[i] and 
              close[i] < ema34_1d_aligned[i] and 
              close[i] < ema20_4h_aligned[i] and 
              vol_spike_1d_aligned[i]):
            signals[i] = -0.20
        else:
            # Hold previous position
            if i > 0:
                signals[i] = signals[i-1]
            else:
                signals[i] = 0.0
    
    return signals