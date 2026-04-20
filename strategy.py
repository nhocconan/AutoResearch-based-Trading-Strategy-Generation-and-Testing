#!/usr/bin/env python3
# 4h_1d_1w_Donchian20_Breakout_Volume_Trend
# Hypothesis: On 4h timeframe, trade Donchian(20) breakouts with volume confirmation and 1d/1w EMA trend filter.
# Uses daily and weekly EMA34 to filter trades in trending markets. Targets 20-30 trades per year.
# Works in bull/bear via trend-aligned breakouts and volatility-based position sizing.

name = "4h_1d_1w_Donchian20_Breakout_Volume_Trend"
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
    
    # Get daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 34 or len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily and weekly EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'])
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    close_1w_series = pd.Series(df_1w['close'])
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF data to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high, volume spike, and price above both EMAs (uptrend)
            if (close[i] > high_20[i] and 
                volume[i] > 1.5 * volume_ma[i] and
                close[i] > ema_34_1d_aligned[i] and
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, volume spike, and price below both EMAs (downtrend)
            elif (close[i] < low_20[i] and 
                  volume[i] > 1.5 * volume_ma[i] and
                  close[i] < ema_34_1d_aligned[i] and
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low or trend reversal (below either EMA)
            if close[i] < low_20[i] or close[i] < ema_34_1d_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high or trend reversal (above either EMA)
            if close[i] > high_20[i] or close[i] > ema_34_1d_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals