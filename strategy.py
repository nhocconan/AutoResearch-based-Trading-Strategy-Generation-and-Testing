#!/usr/bin/env python3
name = "12h_Donchian20_TrendVolume_Regime"
timeframe = "12h"
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
    
    # 1D trend: EMA34
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1D Donchian(20) for breakout signals
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    upper_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    
    # 12H volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    # 1D chop regime: avoid trading in choppy markets
    atr_period = 14
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    sma_high_low = pd.Series((high_1d + low_1d) / 2).rolling(window=atr_period, min_periods=atr_period).mean().values
    sma_high_low_aligned = align_htf_to_ltf(prices, df_1d, sma_high_low)
    
    # Chopping index approximation: high-low range vs ATR
    chop_denom = sma_high_low_aligned
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid div by zero
    chop_value = (atr_along / chop_denom) * 100 if 'atr_along' in locals() else 0
    chop_value = (atr_aligned / chop_denom) * 100
    chop_filter = chop_value < 61.8  # trending when chop < 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(upper_1d_aligned[i]) or np.isnan(lower_1d_aligned[i]) or np.isnan(atr_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian + uptrend + volume + trending regime
            if (close[i] > upper_1d_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                vol_filter[i] and 
                chop_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + downtrend + volume + trending regime
            elif (close[i] < lower_1d_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  vol_filter[i] and 
                  chop_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below lower Donchian OR trend changes
            if close[i] < lower_1d_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above upper Donchian OR trend changes
            if close[i] > upper_1d_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals