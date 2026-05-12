# [Experiment #158801] 4h_Donchian_Breakout_20_Volume_Trend_Filter
# Hypothesis: Donchian breakouts capture breakout momentum; volume confirms strength; trend filter avoids counter-trend trades.
# Works in both bull and bear markets: trend filter (1d EMA50) ensures trades align with higher timeframe trend.
# Target: 20-30 trades/year on 4h timeframe.

name = "4h_Donchian_Breakout_20_Volume_Trend_Filter"
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
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Donchian channels (20-period)
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        upper[i] = np.max(high[i - lookback + 1:i + 1])
        lower[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        vol_ma[i] = np.mean(volume[i - lookback + 1:i + 1])
    
    # Volume spike: current volume > 1.5 * 20-period average
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback - 1, 50)  # Ensure Donchian and trend filter are ready
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper Donchian + volume spike + uptrend (price > EMA50)
            if (close[i] > upper[i] and 
                volume_spike[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian + volume spike + downtrend (price < EMA50)
            elif (close[i] < lower[i] and 
                  volume_spike[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below lower Donchian or trend reversal
            if (close[i] < lower[i] or 
                close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above upper Donchian or trend reversal
            if (close[i] > upper[i] or 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3