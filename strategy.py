#!/usr/bin/env python3
# 12h_1w_1d_donchian_breakout_volume_v2
# Hypothesis: 12h Donchian(20) breakout with volume confirmation and 1d/1w trend filters.
# Long when price breaks above 20-period high with volume > 1.5x average and 1d/1w uptrend.
# Short when price breaks below 20-period low with volume > 1.5x average and 1d/1w downtrend.
# Uses ATR-based stop loss to limit drawdown. Designed for 12-30 trades/year on 12h to avoid fee drag.
# Works in bull/bear via multi-timeframe trend alignment and volume confirmation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_donchian_breakout_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    period = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(period-1, n):
        highest_high[i] = np.max(high[i-period+1:i+1])
        lowest_low[i] = np.min(low[i-period+1:i+1])
    
    # Average volume (20-period)
    avg_volume = np.full(n, np.nan)
    for i in range(period-1, n):
        avg_volume[i] = np.mean(volume[i-period+1:i+1])
    
    # ATR (14-period) for stop loss
    atr_period = 14
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    atr = np.full(n, np.nan)
    for i in range(atr_period-1, n):
        atr[i] = np.mean(tr[i-atr_period+1:i+1])
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(period-1, atr_period-1)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(atr[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian low or ATR stop hit
            if close[i] < lowest_low[i] or close[i] < highest_high[i] - 2.0 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high or ATR stop hit
            if close[i] > highest_high[i] or close[i] > lowest_low[i] + 2.0 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high with volume confirmation and uptrend
            if (close[i] > highest_high[i] and 
                volume[i] > 1.5 * avg_volume[i] and 
                close[i] > ema50_1d_aligned[i] and 
                close[i] > ema50_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low with volume confirmation and downtrend
            elif (close[i] < lowest_low[i] and 
                  volume[i] > 1.5 * avg_volume[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  close[i] < ema50_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals