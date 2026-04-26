#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyTrend_VolumeConfirmation
Hypothesis: 6h timeframe with Donchian(20) breakouts in direction of weekly EMA50 trend, confirmed by volume spike (>2x 20-bar MA). Uses discrete position sizing (0.25) and regime filter (weekly ADX > 20) to avoid whipsaws. Designed for low frequency (target 12-30 trades/year) to minimize fee drag, works in bull/bear via weekly trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Weekly ADX regime filter (ADX > 20 for trending market)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_arr = df_1w['close'].values
    
    plus_dm = np.zeros(len(df_1w))
    minus_dm = np.zeros(len(df_1w))
    tr = np.zeros(len(df_1w))
    
    for i in range(1, len(df_1w)):
        plus_dm[i] = max(high_1w[i] - high_1w[i-1], 0) if high_1w[i] - high_1w[i-1] > high_1w[i-1] - low_1w[i] else 0
        minus_dm[i] = max(high_1w[i-1] - low_1w[i], 0) if high_1w[i-1] - low_1w[i] > high_1w[i] - high_1w[i-1] else 0
        tr[i] = max(high_1w[i] - low_1w[i], abs(high_1w[i] - close_1w_arr[i-1]), abs(low_1w[i] - close_1w_arr[i-1]))
    
    period = 14
    alpha = 1.0 / period
    atr_1w = np.zeros(len(df_1w))
    plus_dm_smooth = np.zeros(len(df_1w))
    minus_dm_smooth = np.zeros(len(df_1w))
    
    if len(df_1w) >= period + 1:
        atr_1w[period] = np.mean(tr[1:period+1])
        plus_dm_smooth[period] = np.sum(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.sum(minus_dm[1:period+1])
    
    for i in range(period+1, len(df_1w)):
        atr_1w[i] = atr_1w[i-1] * (1 - alpha) + alpha * tr[i]
        plus_dm_smooth[i] = plus_dm_smooth[i-1] * (1 - alpha) + alpha * plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i-1] * (1 - alpha) + alpha * minus_dm[i]
    
    plus_di_1w = np.zeros(len(df_1w))
    minus_di_1w = np.zeros(len(df_1w))
    dx_1w = np.zeros(len(df_1w))
    
    for i in range(period, len(df_1w)):
        if atr_1w[i] != 0:
            plus_di_1w[i] = 100 * plus_dm_smooth[i] / atr_1w[i]
            minus_di_1w[i] = 100 * minus_dm_smooth[i] / atr_1w[i]
        dx_1w[i] = 100 * abs(plus_di_1w[i] - minus_di_1w[i]) / (plus_di_1w[i] + minus_di_1w[i]) if (plus_di_1w[i] + minus_di_1w[i]) != 0 else 0
    
    adx_1w = np.zeros(len(df_1w))
    if len(df_1w) >= period*2 + 1:
        adx_1w[period*2] = np.mean(dx_1w[period+1:period*2+1]) if len(dx_1w[period+1:period*2+1]) > 0 else 0
    
    for i in range(period*2+1, len(df_1w)):
        adx_1w[i] = adx_1w[i-1] * (1 - alpha) + alpha * dx_1w[i]
    
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Donchian(20) channels on 6h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size
    
    # Warmup: max of calculations
    start_idx = max(lookback, 20, 50, period*2+1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(adx_1w_aligned[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        hh_val = highest_high[i]
        ll_val = lowest_low[i]
        ema_50_val = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        adx_val = adx_1w_aligned[i]
        
        # Determine weekly trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_1w = close_val > ema_50_val
        bearish_1w = close_val < ema_50_val
        
        # Regime filter: only trade in trending markets (ADX > 20)
        trending_regime = adx_val > 20
        
        # Entry conditions
        long_entry = (close_val > hh_val) and bullish_1w and vol_spike and trending_regime
        short_entry = (close_val < ll_val) and bearish_1w and vol_spike and trending_regime
        
        # Exit conditions: price returns inside Donchian channel or trend reversal or regime change
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and (close_val < hh_val or not bullish_1w or not trending_regime):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close_val > ll_val or not bearish_1w or not trending_regime):
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyTrend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0