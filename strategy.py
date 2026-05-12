#!/usr/bin/env python3
# 4H_DONCHIAN_BREAKOUT_20_VOLUME_CONFIRMATION_1D_TREND_FILTER
# Hypothesis: 4-hour Donchian channel (20-period) breakouts with 1-day trend filter (EMA34) and volume confirmation capture momentum in both bull and bear markets.
# Long when price breaks above upper band in uptrend with above-average volume; short when breaks below lower band in downtrend with above-average volume.
# Uses volume ratio (current volume / 20-period average) > 1.2 for confirmation to reduce false breakouts.
# Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years).

name = "4H_DONCHIAN_BREAKOUT_20_VOLUME_CONFIRMATION_1D_TREND_FILTER"
timeframe = "4h"
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
    
    # 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 4-hour Donchian channel (20-period)
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        upper[i] = np.max(high[i - lookback + 1:i + 1])
        lower[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Volume confirmation: current volume / 20-period average volume
    vol_ma = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        vol_ma[i] = np.mean(volume[i - lookback + 1:i + 1])
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = lookback - 1  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper band in uptrend with volume confirmation
            if (close[i] > upper[i] and 
                close[i] > ema34_1d_aligned[i] and 
                vol_ratio[i] > 1.2):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower band in downtrend with volume confirmation
            elif (close[i] < lower[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  vol_ratio[i] > 1.2):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below lower band or trend reversal
            if (close[i] < lower[i] or 
                close[i] <= ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above upper band or trend reversal
            if (close[i] > upper[i] or 
                close[i] >= ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals