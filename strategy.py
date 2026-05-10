# !/usr/bin/env python3
# 12h_PriceAction_Structure_1dTrend_Volume
# Hypothesis: Price action structure (higher highs/lows for long, lower highs/lows for short) combined with 1d EMA trend filter and volume confirmation. Works in bull via trend continuation and bear via counter-trend swings. Target 12h timeframe for lower trade frequency and higher win rate.

name = "12h_PriceAction_Structure_1dTrend_Volume"
timeframe = "12h"
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
    
    # 1d data for EMA trend and volume average
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d volume 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20_1d = mean_arr(volume_1d, 20)
    
    # Align 1d indicators to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Price action structure: higher highs and higher lows for uptrend, lower highs and lower lows for downtrend
    # We look for consecutive higher highs and higher lows (bullish structure) or consecutive lower highs and lower lows (bearish structure)
    hh = np.zeros(n, dtype=bool)  # higher high
    hl = np.zeros(n, dtype=bool)  # higher low
    lh = np.zeros(n, dtype=bool)  # lower high
    ll = np.zeros(n, dtype=bool)  # lower low
    
    for i in range(1, n):
        hh[i] = high[i] > high[i-1]
        hl[i] = low[i] > low[i-1]
        lh[i] = high[i] < high[i-1]
        ll[i] = low[i] < low[i-1]
    
    # Consecutive structure: need 2 consecutive bars for confirmation
    bull_structure = hh & hl  # current bar makes higher high and higher low
    bear_structure = lh & ll  # current bar makes lower high and lower low
    
    # For entry, we want confirmation of structure over last 2 bars
    bull_confirmed = bull_structure & np.roll(bull_structure, 1)
    bear_confirmed = bear_structure & np.roll(bear_structure, 1)
    # Handle first element
    bull_confirmed[0] = False
    bear_confirmed[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Enough history for indicators and structure
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish price structure, above 1d EMA50, volume above average
            if bull_confirmed[i] and close[i] > ema_50_aligned[i] and volume[i] > vol_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish price structure, below 1d EMA50, volume above average
            elif bear_confirmed[i] and close[i] < ema_50_aligned[i] and volume[i] > vol_ma_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: structure breaks or price crosses below EMA
            if not bull_confirmed[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: structure breaks or price crosses above EMA
            if not bear_confirmed[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals