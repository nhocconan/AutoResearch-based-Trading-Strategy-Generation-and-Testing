#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + 1w EMA50 Trend + Volume Spike + Chop Regime Filter
Hypothesis: Donchian channel breakouts capture strong momentum. 1w EMA50 provides major trend filter. Volume spike confirms institutional interest. Chop regime filter avoids ranging markets. Works in bull/bear via discrete sizing (0.25) and trend alignment. Target: 12-37 trades/year on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA50 and Chop regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    ema_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 1w Chop regime filter: CHOP(14) > 61.8 = range (avoid), < 38.2 = trend (favor)
    def calculate_chop(high_arr, low_arr, close_arr, window=14):
        tr1 = np.abs(high_arr[1:] - low_arr[1:])
        tr2 = np.abs(high_arr[1:] - close_arr[:-1])
        tr3 = np.abs(low_arr[1:] - close_arr[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        atr = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        
        highest_high = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        hhll = highest_high - lowest_low
        
        atr_sum = pd.Series(atr).rolling(window=window, min_periods=window).sum().values
        chop = 100 * np.log10(atr_sum / np.log(10) / hhll)
        chop = np.where(hhll == 0, 50, chop)  # avoid division by zero
        return chop
    
    chop_values = calculate_chop(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop_values)
    
    # Donchian(20) channels from 12h data (using rolling window)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for 1w EMA warmup and Donchian lookback
    start_idx = max(50, lookback, 20)  # EMA50 needs ~50, Donchian 20, vol MA 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Regime filter: only trade when NOT choppy (CHOP < 61.8 = trending)
        not_choppy = chop_aligned[i] < 61.8
        
        # Trend filter: price relative to 1w EMA50
        bullish_bias = curr_close > ema_1w_aligned[i]
        bearish_bias = curr_close < ema_1w_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: Donchian breakout + trend + volume + regime
            # Long: price breaks above upper Donchian AND bullish bias AND volume spike AND not choppy
            long_entry = (curr_high > highest_high[i]) and bullish_bias and vol_spike and not_choppy
            # Short: price breaks below lower Donchian AND bearish bias AND volume spike AND not choppy
            short_entry = (curr_low < lowest_low[i]) and bearish_bias and vol_spike and not_choppy
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below lower Donchian (mean reversion) OR loss of bullish bias OR choppy regime
            if (curr_low < lowest_low[i]) or (curr_close < ema_1w_aligned[i]) or (chop_aligned[i] >= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above upper Donchian (mean reversion) OR loss of bearish bias OR choppy regime
            if (curr_high > highest_high[i]) or (curr_close > ema_1w_aligned[i]) or (chop_aligned[i] >= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0