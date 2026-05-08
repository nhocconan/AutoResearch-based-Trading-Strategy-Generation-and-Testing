#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WilliamsVixFix_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter and VixFix calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data for VixFix calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams VixFix on daily data
    # VixFix = ((Highest Close in period - Low) / Highest Close in period) * 100
    lookback = 22
    highest_close = pd.Series(df_1d['close']).rolling(window=lookback, min_periods=lookback).max().values
    vixfix = ((highest_close - df_1d['low'].values) / highest_close) * 100
    vixfix = np.nan_to_num(vixfix, nan=0.0)
    
    # VixFix moving average for signal generation
    vixfix_ma = pd.Series(vixfix).rolling(window=10, min_periods=10).mean().values
    
    # Weekly EMA for trend filter
    weekly_close = df_1w['close'].values
    ema_20_1w = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation - 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # Align weekly trend and daily indicators to 6h timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    vixfix_ma_aligned = align_htf_to_ltf(prices, df_1d, vixfix_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 200
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(vixfix_ma_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: VixFix spikes high (fear) + above weekly EMA + volume confirmation
            if (vixfix_ma_aligned[i] > 30 and 
                close[i] > ema_20_1w_aligned[i] and
                vol_ratio[i] > 1.8):
                signals[i] = 0.25
                position = 1
            # Short: VixFix spikes high (fear) + below weekly EMA + volume confirmation
            elif (vixfix_ma_aligned[i] > 30 and 
                  close[i] < ema_20_1w_aligned[i] and
                  vol_ratio[i] > 1.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: VixFix drops below threshold OR price crosses below weekly EMA
            if vixfix_ma_aligned[i] < 20 or close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: VixFix drops below threshold OR price crosses above weekly EMA
            if vixfix_ma_aligned[i] < 20 or close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals