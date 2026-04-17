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
    
    # === 1d data for trend context (higher timeframe) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA50 for trend filter (more stable than 20)
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d volume average (20-period) for volume filter
    vol_avg20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg20_1d)
    
    # === 4h data for entry triggers ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # 4h Donchian channels (20-period) for breakouts
    donch_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donch_high_20_aligned = align_htf_to_ltf(prices, df_4h, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_4h, donch_low_20)
    
    # 4h volume average (20-period) for volume confirmation
    vol_avg20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_avg20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_avg20_4h)
    
    signals = np.zeros(n)
    
    # Warmup covers the longest lookback (50 for EMA50)
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(donch_high_20_aligned[i]) or 
            np.isnan(donch_low_20_aligned[i]) or
            np.isnan(vol_avg20_4h_aligned[i]) or
            np.isnan(vol_avg20_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current values
        vol_4h_current = align_htf_to_ltf(prices, df_4h, volume_4h)[i]
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        
        # Volume filters: current volume > 1.5x average on both timeframes
        vol_filter_4h = vol_4h_current > 1.5 * vol_avg20_4h_aligned[i]
        vol_filter_1d = vol_1d_current > 1.3 * vol_avg20_1d_aligned[i]  # Slightly lower for 1d
        
        # Entry conditions
        if position == 0:
            # Long: price breaks above Donchian high + 1d uptrend + volume confirmation
            breakout_up = close[i] > donch_high_20_aligned[i]
            uptrend = close[i] > ema50_1d_aligned[i]
            
            if breakout_up and uptrend and vol_filter_4h and vol_filter_1d:
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: price breaks below Donchian low + 1d downtrend + volume confirmation
            breakout_down = close[i] < donch_low_20_aligned[i]
            downtrend = close[i] < ema50_1d_aligned[i]
            
            if breakout_down and downtrend and vol_filter_4h and vol_filter_1d:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions: opposite breakout or loss of trend
        elif position == 1:
            # Exit long when price breaks below Donchian low or trend turns down
            breakout_down = close[i] < donch_low_20_aligned[i]
            trend_down = close[i] < ema50_1d_aligned[i]
            
            if breakout_down or trend_down:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price breaks above Donchian high or trend turns up
            breakout_up = close[i] > donch_high_20_aligned[i]
            trend_up = close[i] > ema50_1d_aligned[i]
            
            if breakout_up or trend_up:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0