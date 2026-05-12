#!/usr/bin/env python3
# 4H_DONCHIAN_BREAKOUT_20_VOLUME_CONFIRMATION_1D_TREND_FILTER
# Hypothesis: On 4h timeframe, Donchian(20) breakouts with volume confirmation and 1d EMA50 trend filter capture momentum in both bull and bear markets.
# Breakouts above upper band in uptrend (close > EMA50) go long; breakdowns below lower band in downtrend (close < EMA50) go short.
# Volume > 1.5x 20-period average confirms genuine breakout. Reduces false signals and controls trade frequency.
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
    
    # Calculate Donchian channels (20-period)
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        upper[i] = np.max(high[i - lookback + 1:i + 1])
        lower[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Calculate 20-period average volume for confirmation
    vol_ma = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        vol_ma[i] = np.mean(volume[i - lookback + 1:i + 1])
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # EMA50 for trend filter
    ema50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = lookback - 1  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # LONG: Price breaks above upper Donchian band in uptrend
            if (close[i] > upper[i] and 
                close[i] > ema50_aligned[i] and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian band in downtrend
            elif (close[i] < lower[i] and 
                  close[i] < ema50_aligned[i] and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below lower Donchian band or trend reversal
            if (close[i] < lower[i] or 
                close[i] <= ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above upper Donchian band or trend reversal
            if (close[i] > upper[i] or 
                close[i] >= ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals