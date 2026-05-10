#!/usr/bin/env python3
# 6h_LongTerm_Trend_Follow_with_Daily_Filter
# Hypothesis: In crypto markets, strong multi-day trends persist despite short-term noise.
# This strategy uses 60-period EMA on 6h as primary trend filter, confirmed by 1d EMA34 direction.
# Entries occur on pullbacks to the 6h EMA60 with volume confirmation, exploiting the tendency
# of trends to resume after consolidation. Works in both bull and bear markets by following
# the dominant trend on higher timeframe. Low trade frequency expected due to strict
# alignment of 6EMA60 pullback, 1D trend filter, and volume spike requirements.

name = "6h_LongTerm_Trend_Follow_with_Daily_Filter"
timeframe = "6h"
leverage = 1.0

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
    
    # 6h EMA60 for primary trend and dynamic support/resistance
    ema60 = pd.Series(close).ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # 1d EMA34 for higher timeframe trend filter (updated daily)
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)  # aligned to 6h, waits for daily close
    
    # Volume confirmation: 20-period average on 6h
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 20)  # need EMA60 and vol MA ready
    
    for i in range(start_idx, n):
        if np.isnan(ema60[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        price_to_ema = close[i] - ema60[i]
        
        if position == 0:
            # Long setup: price above EMA60 (bullish bias), 1D uptrend, pulling back to EMA60 with volume
            if close[i] > ema60[i] and ema34_1d_aligned[i] > ema34_1d_aligned[i-1] and price_to_ema < 0 and vol_confirm:
                # Enter long on pullback to EMA60 with volume confirmation
                signals[i] = 0.25
                position = 1
            # Short setup: price below EMA60 (bearish bias), 1D downtrend, pulling back to EMA60 with volume
            elif close[i] < ema60[i] and ema34_1d_aligned[i] < ema34_1d_aligned[i-1] and price_to_ema > 0 and vol_confirm:
                # Enter short on pullback to EMA60 with volume confirmation
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below EMA60 OR 1D trend turns down
            if close[i] < ema60[i] or ema34_1d_aligned[i] < ema34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above EMA60 OR 1D trend turns up
            if close[i] > ema60[i] or ema34_1d_aligned[i] > ema34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals