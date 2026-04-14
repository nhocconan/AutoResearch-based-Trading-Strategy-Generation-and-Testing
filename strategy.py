#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly trend (HTF)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 20-period EMA on weekly close
    close_1w_series = pd.Series(close_1w)
    ema_20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly EMA to daily timeframe (wait for weekly bar close)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Get daily data for price action
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily range position (where close is in daily range)
    daily_range = high_1d - low_1d
    range_pos = np.where(daily_range > 0, (close_1d - low_1d) / daily_range, 0.5)
    
    # Volume ratio: current volume vs 20-day average
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume_1d / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 14, 20)  # EMA20, ATR14, VolMA20
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(atr[i]) or
            np.isnan(range_pos[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long conditions:
            # 1. Weekly trend up: price above weekly EMA20
            # 2. Strong close in daily range: close in upper 60% of daily range
            # 3. Above average volume: volume > 1.2x 20-day average
            if (price > ema_20_1w_aligned[i] and 
                range_pos[i] > 0.6 and 
                vol_ratio[i] > 1.2):
                position = 1
                signals[i] = position_size
            # Short conditions:
            # 1. Weekly trend down: price below weekly EMA20
            # 2. Weak close in daily range: close in lower 40% of daily range
            # 3. Above average volume: volume > 1.2x 20-day average
            elif (price < ema_20_1w_aligned[i] and 
                  range_pos[i] < 0.4 and 
                  vol_ratio[i] > 1.2):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: weekly trend turns down OR weak close in lower 40% of range
            if (price < ema_20_1w_aligned[i] or range_pos[i] < 0.4):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: weekly trend turns up OR strong close in upper 60% of range
            if (price > ema_20_1w_aligned[i] or range_pos[i] > 0.6):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_EMA_Trend_Range_Position"
timeframe = "1d"
leverage = 1.0