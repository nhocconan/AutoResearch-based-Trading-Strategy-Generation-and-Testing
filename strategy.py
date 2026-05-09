#!/usr/bin/env python3
# 6h_WeeklyTrend_DailyMeanReversion
# Hypothesis: Weekly trend direction (via weekly close vs 34 EMA) determines bias.
# In uptrend: buy dips to daily 34 EMA with volume confirmation.
# In downtrend: sell rallies to daily 34 EMA with volume confirmation.
# Uses 6h for entry timing, targeting mean reversion within trend.
# Designed to work in both bull and bear markets by trading pullbacks in established weekly trends.
# Target: 15-30 trades/year per symbol with disciplined entries.

name = "6h_WeeklyTrend_DailyMeanReversion"
timeframe = "6h"
leverage = 1.0

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Get daily data for EMA and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Weekly trend: close vs 34 EMA
    ema_34_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 34:
        ema_34_1w[33] = np.mean(close_1w[0:34])
        for i in range(34, len(close_1w)):
            ema_34_1w[i] = (close_1w[i] * 2 + ema_34_1w[i-1] * 33) / 34
    
    weekly_uptrend = close_1w > ema_34_1w  # True when above EMA
    
    # Daily mean reversion target: 34 EMA
    ema_34_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[0:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = (close_1d[i] * 2 + ema_34_1d[i-1] * 33) / 34
    
    # Daily volume average (20-period) for confirmation
    vol_ma_20_1d = np.full_like(volume_1d, np.nan)
    if len(volume_1d) >= 20:
        vol_ma_20_1d[19] = np.mean(volume_1d[0:20])
        for i in range(20, len(volume_1d)):
            vol_ma_20_1d[i] = (volume_1d[i] * 19 + vol_ma_20_1d[i-1]) / 20
    
    # Align all indicators to 6h timeframe
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(weekly_uptrend_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or \
           np.isnan(vol_ma_20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current 6h volume > 1.5x daily average volume
        # Approximate daily volume from 6h: 4 bars per day, so scale accordingly
        vol_threshold = vol_ma_20_1d_aligned[i] * 1.5 / 4.0  # Scale daily avg to per 6h bar
        volume_ok = volume[i] > vol_threshold
        
        if position == 0:
            # In weekly uptrend: look for long entries on dips to daily EMA
            if weekly_uptrend_aligned[i] > 0.5:  # Weekly uptrend
                # Buy when price touches or dips slightly below daily EMA with volume
                if close[i] <= ema_34_1d_aligned[i] * 1.002 and volume_ok:  # Within 0.2% above/below
                    signals[i] = 0.25
                    position = 1
            # In weekly downtrend: look for short entries on rallies to daily EMA
            else:  # Weekly downtrend
                # Sell when price touches or rallies slightly above daily EMA with volume
                if close[i] >= ema_34_1d_aligned[i] * 0.998 and volume_ok:  # Within 0.2% above/below
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: price moves significantly above EMA (trend resumption) or weekly trend fails
            if close[i] > ema_34_1d_aligned[i] * 1.015 or weekly_uptrend_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price moves significantly below EMA or weekly trend fails
            if close[i] < ema_34_1d_aligned[i] * 0.985 or weekly_uptrend_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals