#!/usr/bin/env python3
# 12h_Donchian20_Breakout_1dTrend_VolumeConfirmation
# Hypothesis: Breakout above/below 20-bar Donchian channel on 12h timeframe with 1d trend filter and volume spike confirmation.
# In bull markets: breakout above upper band triggers long; in bear markets: breakout below lower band triggers short.
# Trend filter uses 1d EMA50 to avoid counter-trend trades. Volume confirmation ensures breakout legitimacy.
# Targets ~20-40 trades/year to minimize fee drag and avoid overtrading.

name = "12h_Donchian20_Breakout_1dTrend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h Donchian channel (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = close_1d > ema50_1d
    trend_1d_down = close_1d < ema50_1d
    
    # Align 1d trend to 12h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(volume_confirmation[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above upper Donchian band in uptrend with volume confirmation
            if (high[i] > highest_high[i] and
                trend_1d_up_aligned[i] > 0.5 and
                volume_confirmation[i]):
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower Donchian band in downtrend with volume confirmation
            elif (low[i] < lowest_low[i] and
                  trend_1d_down_aligned[i] > 0.5 and
                  volume_confirmation[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below middle of Donchian channel or trend reversal
            mid_band = (highest_high[i] + lowest_low[i]) / 2
            if (close[i] < mid_band or
                trend_1d_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above middle of Donchian channel or trend reversal
            mid_band = (highest_high[i] + lowest_low[i]) / 2
            if (close[i] > mid_band or
                trend_1d_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals