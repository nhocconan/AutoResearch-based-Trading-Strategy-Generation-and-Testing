#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1d trend filter and volume confirmation
# - 1d EMA(34) defines trend direction (long when price > EMA34, short when price < EMA34)
# - 6h price closes above/below 1d EMA34 with volume > 1.5x 20-period average for entry
# - Exit when price crosses back below/above 1d EMA34 or on opposite momentum
# - Position size: 0.25 (25%) to manage drawdown
# - Designed to work in both bull and bear markets by following higher timeframe trend
# - Target: 15-30 trades/year to avoid excessive fee drift

name = "6h_EMA34_1dTrend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and volume average
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(34) for trend direction
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 6h volume > 1.5x 1d average volume
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Look for long entry: price above 1d EMA34 + volume confirmation
            if close[i] > ema_34_1d_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: price below 1d EMA34 + volume confirmation
            elif close[i] < ema_34_1d_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price crosses below 1d EMA34
            if close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price crosses above 1d EMA34
            if close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals