#!/usr/bin/env python3
name = "6h_ChaikinMoneyFlow_Pullback_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and CMF
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d Chaikin Money Flow (20-period) for accumulation/distribution
    mf_multiplier = np.where((df_1d['high'] - df_1d['low']) != 0, 
                             ((df_1d['close'] - df_1d['low']) - (df_1d['high'] - df_1d['close'])) / (df_1d['high'] - df_1d['low']), 
                             0)
    mf_volume = mf_multiplier * df_1d['volume'].values
    cmf_20 = pd.Series(mf_volume).rolling(window=20, min_periods=20).sum() / \
             pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).sum()
    cmf_20_values = cmf_20.values
    cmf_20_aligned = align_htf_to_ltf(prices, df_1d, cmf_20_values)
    
    # 6h volume filter: current volume > 2.0x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 4  # ~1 day for 6h to reduce trades
    
    start_idx = max(100, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(cmf_20_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine 1d trend direction
        trend_up = close > ema_50_1d_aligned[i]
        trend_down = close < ema_50_1d_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price pulls back to EMA50 in uptrend with positive CMF and volume spike
            if (low[i] <= ema_50_1d_aligned[i] <= high[i] and 
                trend_up[i] and 
                cmf_20_aligned[i] > 0.05 and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price pulls back to EMA50 in downtrend with negative CMF and volume spike
            elif (low[i] <= ema_50_1d_aligned[i] <= high[i] and 
                  trend_down[i] and 
                  cmf_20_aligned[i] < -0.05 and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price closes below EMA50 or CMF turns negative
            if close[i] < ema_50_1d_aligned[i] or cmf_20_aligned[i] < -0.05:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price closes above EMA50 or CMF turns positive
            if close[i] > ema_50_1d_aligned[i] or cmf_20_aligned[i] > 0.05:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Chaikin Money Flow pullback strategy on 6h timeframe.
# Long when price pulls back to 1d EMA50 during uptrend with CMF > 0.05 (accumulation) and volume spike.
# Short when price pulls back to 1d EMA50 during downtrend with CMF < -0.05 (distribution) and volume spike.
# Uses 1d trend filter and CMF for institutional flow confirmation. Works in both bull (buy pullbacks) and bear (sell rallies).
# Volume spike ensures institutional participation. Target: 50-150 total trades over 4 years (12-37/year).
# Based on successful volume+trend patterns from DB, adding CMF as a novel institutional flow filter.