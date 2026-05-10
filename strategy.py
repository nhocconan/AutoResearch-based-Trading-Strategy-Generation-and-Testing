#!/usr/bin/env python3
# 1d_Williams_VIX_Fix_1wTrend_VolumeFilter
# Hypothesis: Williams VIX Fix (synthetic volatility) combined with 1-week EMA trend filter and volume confirmation.
# VIX Fix measures market fear; extreme readings signal potential reversals. Works in both bull and bear markets
# by aligning with higher timeframe trend. Uses volume spike for confirmation and targets 15-25 trades/year.
# Williams VIX Fix formula: ((HighestClose - Low) / (HighestClose - HighestClose)) * 100, where HighestClose is
# the highest close over the lookback period. Values > 80 indicate extreme fear (potential bottom).

name = "1d_Williams_VIX_Fix_1wTrend_VolumeFilter"
timeframe = "1d"
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
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Williams VIX Fix with 22-day lookback (approx 1 month)
    lookback = 22
    highest_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).max().values
    vix_fix = ((highest_close - low) / (highest_close - highest_close + 1e-10)) * 100  # Avoid division by zero
    vix_fix = np.where(highest_close == highest_close, 100, vix_fix)  # When close == highest_close
    
    # Get weekly EMA for trend filter
    ema_21_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 21, 20)  # Warmup for VIX Fix, weekly EMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(vix_fix[i]) or np.isnan(ema_21_1w_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_21_1w_aligned[i]
        downtrend = close[i] < ema_21_1w_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # VIX Fix signals: extreme fear (>80) = potential long, low fear (<20) = potential short
        extreme_fear = vix_fix[i] > 80
        low_fear = vix_fix[i] < 20
        
        if position == 0:
            # Long entry: extreme fear + uptrend + volume spike
            if extreme_fear and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: low fear + downtrend + volume spike
            elif low_fear and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: fear subsides or trend reversal
            if vix_fix[i] < 50 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: fear increases or trend reversal
            if vix_fix[i] > 50 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals