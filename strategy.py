#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for trend context ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d 14-period ATR for volatility filter
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), np.abs(high_1d[1:] - close_1d[:-1]))
    tr = np.concatenate([[np.inf], np.maximum(tr1, tr2)])
    atr14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # === 12h data for entry triggers ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # 12h Donchian channels (20-period) for breakouts
    donch_high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donch_high_20_aligned = align_htf_to_ltf(prices, df_12h, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_12h, donch_low_20)
    
    # 12h volume average (20-period) for volume confirmation
    vol_avg20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_avg20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg20_12h)
    
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
            np.isnan(vol_avg20_12h_aligned[i]) or
            np.isnan(atr14_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 12h values
        vol_12h_current = align_htf_to_ltf(prices, df_12h, volume_12h)[i]
        
        # Volume filter: current volume > 1.8x average
        vol_filter = vol_12h_current > 1.8 * vol_avg20_12h_aligned[i]
        
        # Entry conditions
        if position == 0:
            # Long: price breaks above Donchian high + 1d uptrend + volume confirmation
            breakout_up = close[i] > donch_high_20_aligned[i]
            uptrend = close[i] > ema50_1d_aligned[i]
            
            if breakout_up and uptrend and vol_filter:
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: price breaks below Donchian low + 1d downtrend + volume confirmation
            breakout_down = close[i] < donch_low_20_aligned[i]
            downtrend = close[i] < ema50_1d_aligned[i]
            
            if breakout_down and downtrend and vol_filter:
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

name = "12h_Donchian_Breakout_1dTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0