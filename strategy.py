#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d trend filter and weekly volume confirmation
# - 1d EMA(34) defines trend direction (long when price > EMA34, short when price < EMA34)
# - Weekly volume > 1.3x 4-week average for conviction
# - 12h Williams %R(14) for entry timing: long when %R < -80 in uptrend, short when %R > -20 in downtrend
# - Exit on opposite %R extreme (%R > -20 for long, %R < -80 for short) or trend reversal
# - Position size: 0.25 (25%) to manage drawdown
# - Designed to work in both bull and bear markets by following higher timeframe trend
# - Target: 15-30 trades/year to avoid excessive fee drift

name = "12h_EMA34_WilliamsR_WeeklyVolume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(34) for trend direction
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get weekly data for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly volume average (4-period)
    vol_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(vol_1w).rolling(window=4, min_periods=4).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # 12h Williams %R(14) for entry timing
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan)
    williams_r_values = williams_r.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_1w_aligned[i]) or 
            np.isnan(williams_r_values[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current weekly volume > 1.3x average
        volume_filter = vol_ma_1w_aligned[i] > 0 and volume[i] > 1.3 * vol_ma_1w_aligned[i]
        
        if position == 0:
            # Look for long entry: uptrend (price > 1d EMA34) + oversold Williams %R + volume
            if close[i] > ema_34_1d_aligned[i] and williams_r_values[i] < -80 and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: downtrend (price < 1d EMA34) + overbought Williams %R + volume
            elif close[i] < ema_34_1d_aligned[i] and williams_r_values[i] > -20 and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on overbought Williams %R or trend reversal
            if williams_r_values[i] > -20 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on oversold Williams %R or trend reversal
            if williams_r_values[i] < -80 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals