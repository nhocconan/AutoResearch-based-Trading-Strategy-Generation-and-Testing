#!/usr/bin/env python3
# 12h_1W_Trend_Retracement_With_1D_Volume
# Hypothesis: Use 1-week trend direction, enter retracement to 1-day EMA34 on 12h closes with volume confirmation.
# Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend). Low frequency: ~20-40 trades/year.
# Target: 12h timeframe, weekly trend filter, daily EMA retracement, volume spike.

name = "12h_1W_Trend_Retracement_With_1D_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-week trend: EMA50
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1-day EMA34 for retracement entries
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1-day volume average (20-period)
    df_1d_vol = df_1d['volume'].values
    vol_avg_20 = pd.Series(df_1d_vol).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 70  # Ensure warmup for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current 12h volume > 1.8x daily average volume
        volume_condition = volume[i] > 1.8 * vol_avg_20_aligned[i]
        
        if position == 0:
            # Long: uptrend (1w EMA50 up) + price near 1d EMA34 (retracement) + volume spike
            if ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]:  # 1w uptrend
                if low[i] <= ema_34_1d_aligned[i] * 1.005 and high[i] >= ema_34_1d_aligned[i] * 0.995:  # near EMA34
                    if volume_condition:
                        signals[i] = 0.25
                        position = 1
            # Short: downtrend (1w EMA50 down) + price near 1d EMA34 (retracement) + volume spike
            elif ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]:  # 1w downtrend
                if low[i] <= ema_34_1d_aligned[i] * 1.005 and high[i] >= ema_34_1d_aligned[i] * 0.995:  # near EMA34
                    if volume_condition:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: trend turns down or price moves above 1d EMA34 by 1%
            if ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1] or close[i] > ema_34_1d_aligned[i] * 1.01:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend turns up or price moves below 1d EMA34 by 1%
            if ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1] or close[i] < ema_34_1d_aligned[i] * 0.99:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals