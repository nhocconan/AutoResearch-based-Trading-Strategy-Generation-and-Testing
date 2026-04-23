#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d EMA34 trend + volume confirmation + ATR trailing stop
Donchian channels capture volatility-based breakouts. EMA34 on 1d provides reliable trend filter.
Volume spike confirms institutional interest. ATR trailing stop manages risk. 4h timeframe balances
signal quality and trade frequency. Works in bull markets (breakouts with trend) and bear markets
(short breakdowns with trend). Target: 25-40 trades/year (100-160 over 4 years) with discrete sizing 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for volatility and trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)  # need EMA34_1d, Donchian, ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_since_long = 0.0
                lowest_since_short = 0.0
            continue
        
        if position == 0:
            # Long: Break above Donchian upper AND price > 1d EMA34 (uptrend) AND volume spike
            if (close[i] > highest_high[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.8 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
                highest_since_long = high[i]
            # Short: Break below Donchian lower AND price < 1d EMA34 (downtrend) AND volume spike
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.8 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
                lowest_since_short = low[i]
        else:
            # Update highest/lowest since position entry
            if position == 1:
                highest_since_long = max(highest_since_long, high[i])
                # Exit: ATR trailing stop OR trend reversal
                if close[i] < highest_since_long - 2.5 * atr[i] or close[i] < ema_34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    highest_since_long = 0.0
                else:
                    signals[i] = 0.25
            elif position == -1:
                lowest_since_short = min(lowest_since_short, low[i])
                # Exit: ATR trailing stop OR trend reversal
                if close[i] > lowest_since_short + 2.5 * atr[i] or close[i] > ema_34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    lowest_since_short = 0.0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4H_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_ATRTrail"
timeframe = "4h"
leverage = 1.0