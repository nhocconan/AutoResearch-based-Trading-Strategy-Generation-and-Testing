#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ADX filter and volume confirmation
# Uses standard Donchian period for balanced trade frequency.
# 1d ADX > 20 filters for trending markets only (avoids chop).
# Volume > 1.5x median ensures participation.
# Designed to work in both bull and bear markets by only trading when trend is present.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # 1d ADX(14) for trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    
    # +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM
    tr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_14 = adx  # Already smoothed
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Volume confirmation: current > 1.5x median of last 50 bars
    vol_median = pd.Series(volume).rolling(window=50, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(adx_14_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Trend filter: ADX > 20 indicates trending market
        trend_filter = adx_14_aligned[i] > 20
        
        # Long: Donchian breakout up + volume + trend
        if (close[i] > high_20[i] and 
            volume[i] > vol_threshold[i] and 
            trend_filter):
            signals[i] = 0.25
        
        # Short: Donchian breakout down + volume + trend
        elif (close[i] < low_20[i] and 
              volume[i] > vol_threshold[i] and 
              trend_filter):
            signals[i] = -0.25
        
        # Exit: price re-enters Donchian channel
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < high_20[i]) or
               (signals[i-1] == -0.25 and close[i] > low_20[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_DonchianBreakout20_ADX20_Volume1.5x"
timeframe = "4h"
leverage = 1.0