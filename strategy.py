#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WilliamsVixFix_1wTrend_Volume"
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
    
    # Get weekly data for VixFix and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Williams VixFix on weekly data
    # VixFix = (Highest Close in period - Low) / Highest Close in period * 100
    lookback = 22  # approximately 1 month of weekly data
    highest_close = pd.Series(df_1w['close'].values).rolling(window=lookback, min_periods=lookback).max().values
    vixfix = (highest_close - df_1w['low'].values) / highest_close * 100
    vixfix = np.nan_to_num(vixfix, nan=0.0)
    
    # VixFix moving average for signal generation
    vixfix_ma = pd.Series(vixfix).rolling(window=10, min_periods=10).mean().values
    
    # Weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly indicators to 6h timeframe
    vixfix_ma_aligned = align_htf_to_ltf(prices, df_1w, vixfix_ma)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation - 24-period average volume (6h, equivalent to 6 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(vixfix_ma_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: VixFix spikes above MA (fear spike) + above weekly EMA34 + volume confirmation
            if (vixfix_ma_aligned[i] > vixfix_ma_aligned[i-1] and  # VixFix rising
                close[i] > ema_34_1w_aligned[i] and                # Above weekly trend
                vol_ratio[i] > 2.0):                               # High volume
                signals[i] = 0.25
                position = 1
            # Short: VixFix drops below MA (fear subsiding) + below weekly EMA34 + volume confirmation
            elif (vixfix_ma_aligned[i] < vixfix_ma_aligned[i-1] and  # VixFix falling
                  close[i] < ema_34_1w_aligned[i] and               # Below weekly trend
                  vol_ratio[i] > 2.0):                              # High volume
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: VixFix falls back below MA OR price drops below weekly EMA34
            if vixfix_ma_aligned[i] < vixfix_ma_aligned[i-1] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: VixFix rises back above MA OR price rises above weekly EMA34
            if vixfix_ma_aligned[i] > vixfix_ma_aligned[i-1] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals