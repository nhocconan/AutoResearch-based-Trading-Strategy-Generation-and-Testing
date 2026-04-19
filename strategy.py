#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h trend filter and volume confirmation
# - 12h EMA(34) defines trend direction (long when price > EMA34, short when price < EMA34)
# - 4h Donchian(20) breakout for entry: long when price > upper band, short when price < lower band
# - 1d volume > 1.8x 20-period average for conviction
# - Exit on opposite Donchian band touch or trend reversal
# - Position size: 0.25 (25%) to manage drawdown
# - Designed to work in both bull and bear markets by following higher timeframe trend
# - Target: 20-40 trades/year to avoid excessive fee drift

name = "4h_Donchian20_12hTrend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA(34) for trend direction
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 4h Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(high_20[i]) or 
            np.isnan(low_20[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 1d volume > 1.8x average
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.8 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Look for long entry: uptrend (price > 12h EMA34) + Donchian breakout + volume
            if close[i] > ema_34_12h_aligned[i] and close[i] > high_20[i-1] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: downtrend (price < 12h EMA34) + Donchian breakdown + volume
            elif close[i] < ema_34_12h_aligned[i] and close[i] < low_20[i-1] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on Donchian lower band touch or trend reversal
            if close[i] < low_20[i] or close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on Donchian upper band touch or trend reversal
            if close[i] > high_20[i] or close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals