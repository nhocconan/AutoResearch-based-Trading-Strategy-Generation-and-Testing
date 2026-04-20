#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly ATR (14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1w[0] = tr1[0]
    
    # Wilder's smoothing
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        alpha = 1.0 / period
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1w = wilder_smooth(tr_1w, 14)
    atr_1w_avg = pd.Series(atr_1w).rolling(window=4, min_periods=4).mean().values  # 4-week average ATR
    atr_1w_avg_aligned = align_htf_to_ltf(prices, df_1w, atr_1w_avg)
    
    # Calculate daily Donchian channels (20)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    donch_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    
    # Calculate daily average volume (20)
    volume_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = prices['close'].iloc[i]
        high_val = prices['high'].iloc[i]
        low_val = prices['low'].iloc[i]
        vol_val = prices['volume'].iloc[i]
        atr_1w_avg_val = atr_1w_avg_aligned[i]
        donch_high_val = donch_high_1d_aligned[i]
        donch_low_val = donch_low_1d_aligned[i]
        vol_avg_val = vol_avg_20_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(atr_1w_avg_val) or np.isnan(donch_high_val) or 
            np.isnan(donch_low_val) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above daily Donchian high with volume surge and volatility expansion
            if (close_val > donch_high_val and 
                vol_val > 2.0 * vol_avg_val and 
                (high_val - low_val) > 1.5 * atr_1w_avg_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily Donchian low with volume surge and volatility expansion
            elif (close_val < donch_low_val and 
                  vol_val > 2.0 * vol_avg_val and 
                  (high_val - low_val) > 1.5 * atr_1w_avg_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below daily Donchian low or volatility contraction
            if close_val < donch_low_val or (high_val - low_val) < 0.5 * atr_1w_avg_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above daily Donchian high or volatility contraction
            if close_val > donch_high_val or (high_val - low_val) < 0.5 * atr_1w_avg_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 6h_Volatility_Volume_Donchian_Breakout_v1
# Uses weekly ATR (4-week average) for volatility filter
# Uses daily Donchian(20) breakouts for entry
# Requires volume > 2x 20-day average and bar range > 1.5x weekly ATR
# Session filter: 8-20 UTC to avoid low-volume periods
# Exits when price breaks opposite Donchian level or volatility contracts
# Designed for 6h timeframe with ~15-30 trades/year
name = "6h_Volatility_Volume_Donchian_Breakout_v1"
timeframe = "6h"
leverage = 1.0