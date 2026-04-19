#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R with 1w trend filter and volume confirmation
# - Williams %R(14) on 1d for mean reversion: long when <-80, short when >-20
# - 1w EMA(50) trend filter: only trade in direction of higher timeframe trend
# - 1d volume > 1.5x 20-period average for conviction
# - Designed to work in both bull and bear markets by following weekly trend
# - Target: 10-20 trades/year to minimize fee drag on daily timeframe

name = "1d_WilliamsR_1wTrend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and volume
    df_1d = get_htf_data(prices, '1d')
    
    # Williams %R(14) on 1d
    highest_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min()
    denominator = highest_high - lowest_low
    williams_r = np.where(denominator != 0, -100 * (highest_high - df_1d['close'].values) / denominator, -50)
    williams_r = np.where(np.isnan(williams_r), -50, williams_r)
    
    # 1d volume average (20-period)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA(50) for trend direction
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF data to 1d
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(williams_r_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 1d volume > 1.5x 20-period average
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Look for long entry: uptrend (price > 1w EMA50) + oversold Williams %R + volume
            if close[i] > ema_50_1w_aligned[i] and williams_r_aligned[i] < -80 and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: downtrend (price < 1w EMA50) + overbought Williams %R + volume
            elif close[i] < ema_50_1w_aligned[i] and williams_r_aligned[i] > -20 and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on overbought Williams %R or trend reversal
            if williams_r_aligned[i] > -20 or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on oversold Williams %R or trend reversal
            if williams_r_aligned[i] < -80 or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals