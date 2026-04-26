#!/usr/bin/env python3
"""
1h_HTFTrend_VolumeSpike_CamarillaBreakout_v1
Hypothesis: Trade 1h Camarilla R1/S1 breakouts only in direction of 4h EMA50 trend with volume spike confirmation.
Uses 1d EMA34 as secondary trend filter to avoid counter-trend trades in strong daily trends.
Designed for 60-150 total trades over 4 years (15-37/year) by requiring confluence of:
1. 4h EMA50 trend (primary direction)
2. 1d EMA34 trend (secondary filter)
3. 1h Camarilla R1/S1 breakout
4. Volume spike (>1.5x 20-period average)
5. Session filter (08-20 UTC)
Uses discrete position sizing (0.20) to minimize fee churn. Works in bull/bear via trend filters:
- Only long breakouts when both 4h and 1d are uptrending
- Only short breakdowns when both 4h and 1d are downtrending
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) - avoids datetime64 TypeError
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for primary trend
    df_4h = get_htf_data(prices, '4h')
    # Calculate 4h EMA50 for primary trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    htf_trend_4h = np.where(close > ema_50_4h_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Load 1d data ONCE before loop for secondary trend filter
    df_1d = get_htf_data(prices, '1d')
    # Calculate 1d EMA34 for secondary trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    htf_trend_1d = np.where(close > ema_34_1d_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate Camarilla pivot levels from 1d data
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    R1_1d = typical_price_1d + (1.1/12) * (df_1d['high'] - df_1d['low'])  # R1 level
    S1_1d = typical_price_1d - (1.1/12) * (df_1d['high'] - df_1d['low'])  # S1 level
    
    # Align Camarilla levels to 1h timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d.values)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d.values)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 4h EMA, 34 for 1d EMA, 20 for volume MA)
    start_idx = max(50, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            # Hold current position or go flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(R1_1d_aligned[i]) or
            np.isnan(S1_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Volume spike condition (1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Only trade when both timeframes agree on trend
        trend_agree_long = (htf_trend_4h[i] == 1 and htf_trend_1d[i] == 1)
        trend_agree_short = (htf_trend_4h[i] == -1 and htf_trend_1d[i] == -1)
        
        # Long breakout above R1 with volume spike and trend agreement
        if trend_agree_long and close[i] > R1_1d_aligned[i] and volume_spike:
            if position != 1:
                signals[i] = 0.20
                position = 1
            else:
                signals[i] = 0.20
        # Short breakdown below S1 with volume spike and trend agreement
        elif trend_agree_short and close[i] < S1_1d_aligned[i] and volume_spike:
            if position != -1:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = -0.20
        # Exit conditions: reverse trend agreement or loss of volume momentum
        elif position == 1 and (not trend_agree_long or close[i] < S1_1d_aligned[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not trend_agree_short or close[i] > R1_1d_aligned[i]):
            signals[i] = 0.0
            position = 0
        # Hold current position
        else:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_HTFTrend_VolumeSpike_CamarillaBreakout_v1"
timeframe = "1h"
leverage = 1.0