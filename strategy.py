#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for indicators
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align daily ATR14 to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate daily ATR(7) for breakout sensitivity
    tr1_7 = high_1d - low_1d
    tr2_7 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_7 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_7[0] = high_1d[0] - low_1d[0]
    tr2_7[0] = np.abs(high_1d[0] - close_1d[0])
    tr3_7[0] = np.abs(low_1d[0] - close_1d[0])
    tr_7 = np.maximum(tr1_7, np.maximum(tr2_7, tr3_7))
    atr_7_1d = pd.Series(tr_7).ewm(alpha=1/7, adjust=False, min_periods=7).mean().values
    atr_7_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_7_1d)
    
    # Calculate daily Donchian channel (20-period)
    donch_high_20 = np.full(len(close_1d), np.nan)
    donch_low_20 = np.full(len(close_1d), np.nan)
    for i in range(20, len(close_1d)):
        donch_high_20[i] = np.max(high_1d[i-20:i])
        donch_low_20[i] = np.min(low_1d[i-20:i])
    
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    
    # Calculate volume moving average (20-period) for daily volume
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # need daily Donchian(20), volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(donch_high_20_aligned[i]) or 
            np.isnan(donch_low_20_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current daily volume > 1.8 * 20-period average
        vol_confirmed = volume_1d_aligned[i] > 1.8 * vol_ma_1d_aligned[i]
        
        # Volatility filter: ATR14 > ATR7 (expanding volatility)
        vol_expanding = atr_1d_aligned[i] > atr_7_1d_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above daily Donchian high with volume and volatility expansion
            if (close[i] > donch_high_20_aligned[i] and 
                vol_confirmed and 
                vol_expanding):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below daily Donchian low with volume and volatility expansion
            elif (close[i] < donch_low_20_aligned[i] and 
                  vol_confirmed and 
                  vol_expanding):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below daily Donchian low or volatility contraction
            if close[i] < donch_low_20_aligned[i] or not vol_expanding:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above daily Donchian high or volatility contraction
            if close[i] > donch_high_20_aligned[i] or not vol_expanding:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeVolatility_Expansion"
timeframe = "4h"
leverage = 1.0