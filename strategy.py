#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX Trend Strength + 1d Williams %R Overbought/Oversold + Volume Spike
# Long when ADX > 25 (trending) and Williams %R < -80 (oversold) with volume confirmation
# Short when ADX > 25 (trending) and Williams %R > -20 (overbought) with volume confirmation
# Uses ADX to filter for trending markets only, avoiding whipsaws in ranging conditions
# Williams %R provides mean reversion entries within the trend
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag
# Designed to work in both bull and bear markets by following established trends

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ADX(14) on 6h data for trend strength
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate Directional Movement
    up_move = high - np.concatenate([[high[0]], high[:-1]])
    down_move = np.concatenate([[low[0]], low[:-1]]) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    plus_dm_sum = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum()
    minus_dm_sum = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_sum / (tr_sum + 1e-10)
    minus_di = 100 * minus_dm_sum / (tr_sum + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Williams %R(14) on 1d data for overbought/oversold levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    
    # Williams %R formula: (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low + 1e-10)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r.values)
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(adx[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_threshold[i])):
            continue
        
        # Long: ADX > 25 (trending), Williams %R < -80 (oversold), volume spike
        if (adx[i] > 25 and 
            williams_r_aligned[i] < -80 and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: ADX > 25 (trending), Williams %R > -20 (overbought), volume spike
        elif (adx[i] > 25 and 
              williams_r_aligned[i] > -20 and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: ADX falls below 20 (losing trend) or Williams %R returns to neutral range
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (adx[i] <= 20 or williams_r_aligned[i] >= -50)) or
               (signals[i-1] == -0.25 and (adx[i] <= 20 or williams_r_aligned[i] <= -50)))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_ADX_WilliamsR_Volume"
timeframe = "6h"
leverage = 1.0