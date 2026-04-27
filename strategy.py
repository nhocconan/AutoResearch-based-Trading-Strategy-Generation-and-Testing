#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for weekly trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Get 1d data for Donchian breakout levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 12h EMA(50) for weekly trend filter (weekly = ~2x 12h bars)
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 1d Donchian(20) breakout levels (based on previous day)
    donch_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().shift(1).values
    donch_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().shift(1).values
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # 6h volume average (20-period) for volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 12h EMA50 and 1d Donchian data
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        ema_trend = ema50_12h_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_filter = vol_current > (vol_ma_val * 1.5)
        
        if position == 0:
            # Long: price breaks above 1d Donchian high with 12h uptrend and volume
            if close[i] > donch_high_aligned[i] and close[i] > ema_trend and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below 1d Donchian low with 12h downtrend and volume
            elif close[i] < donch_low_aligned[i] and close[i] < ema_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 1d Donchian low or 12h trend turns down
            if close[i] < donch_low_aligned[i] or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above 1d Donchian high or 12h trend turns up
            if close[i] > donch_high_aligned[i] or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_12hTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0