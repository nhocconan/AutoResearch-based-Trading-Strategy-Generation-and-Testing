#!/usr/bin/env python3
# 6h_MarketStructure_Breakout_1dTrend_Volume
# Hypothesis: 6h market structure (higher highs/lows) combined with 1d trend filter and volume confirmation.
# Works in bull/bear: Trend filter ensures we trade with higher timeframe momentum, volume confirms breakout strength.
# Market structure filters out false breakouts in ranging markets, focusing on institutional-grade moves.

name = "6h_MarketStructure_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    ema_34_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[0:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = (ema_34_1d[i-1] * 33 + close_1d[i]) / 34
    
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    # Market structure: higher highs and higher lows for uptrend, lower lows and lower highs for downtrend
    # We'll use a simple 5-period structure check
    lookback = 5
    hh_condition = np.zeros(n, dtype=bool)  # higher high
    ll_condition = np.zeros(n, dtype=bool)  # lower low
    
    for i in range(lookback, n):
        # Higher high: current high > highest high in lookback period
        hh_condition[i] = high[i] == np.max(high[i-lookback:i+1])
        # Lower low: current low < lowest low in lookback period
        ll_condition[i] = low[i] == np.min(low[i-lookback:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 20)  # Ensure volume MA is ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: higher high + uptrend (close > EMA34) + volume spike
            if hh_condition[i] and close[i] > ema_34_1d_aligned[i] and volume_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: lower low + downtrend (close < EMA34) + volume spike
            elif ll_condition[i] and close[i] < ema_34_1d_aligned[i] and volume_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: lower low OR trend reversal (close < EMA34)
            if ll_condition[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: higher high OR trend reversal (close > EMA34)
            if hh_condition[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals