#!/usr/bin/env python3
# 12h_1d_1w_donchian_breakout_volume_regime_v1
# Hypothesis: 12h Donchian(20) breakout with volume confirmation and 1d chop regime filter.
# Long when price breaks above Donchian upper band with volume > 1.5x average and chop > 61.8 (range).
# Short when price breaks below Donchian lower band with volume > 1.5x average and chop > 61.8.
# Uses 1w EMA50 as trend filter to avoid counter-trend trades in strong weekly trends.
# Designed for 12-30 trades/year on 12h to avoid fee drag. Works in bull/bear via multi-timeframe alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_donchian_breakout_volume_regime_v1"
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
    period20_high = np.full(n, np.nan)
    period20_low = np.full(n, np.nan)
    for i in range(20, n):
        period20_high[i] = np.max(high[i-20:i+1])
        period20_low[i] = np.min(low[i-20:i+1])
    
    # Average volume (20-period)
    vol_sum = np.zeros(n)
    for i in range(20, n):
        vol_sum[i] = np.sum(volume[i-20:i+1])
    avg_volume = vol_sum / 20
    
    # Get 1d data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr_1d = np.zeros(len(df_1d))
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr_1d[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    # ATR(14) for 1d
    atr_1d = np.zeros(len(df_1d))
    for i in range(14, len(df_1d)):
        atr_1d[i] = np.mean(tr_1d[i-14:i+1])
    
    # Choppiness Index (14-period)
    chop_1d = np.zeros(len(df_1d))
    for i in range(14, len(df_1d)):
        atr_sum = np.sum(atr_1d[i-14:i+1])
        max_high = np.max(high_1d[i-14:i+1])
        min_low = np.min(low_1d[i-14:i+1])
        if max_high != min_low:
            chop_1d[i] = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
        else:
            chop_1d[i] = 50  # neutral when no range
    
    # Align 1d chop to 12h
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(20, 14)  # Ensure Donchian and chop are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(period20_high[i]) or np.isnan(period20_low[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or
            np.isnan(avg_volume[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = volume[i] > 1.5 * avg_volume[i]
        
        # Chop regime: range-bound market (chop > 61.8)
        chop_regime = chop_1d_aligned[i] > 61.8
        
        # 1w trend filter
        uptrend_1w = close[i] > ema50_1w_aligned[i]
        downtrend_1w = close[i] < ema50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian lower band
            if close[i] < period20_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper band
            if close[i] > period20_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian upper band with volume and chop confirmation
            if (close[i] > period20_high[i] and 
                vol_confirm and 
                chop_regime and 
                uptrend_1w):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian lower band with volume and chop confirmation
            elif (close[i] < period20_low[i] and 
                  vol_confirm and 
                  chop_regime and 
                  downtrend_1w):
                position = -1
                signals[i] = -0.25
    
    return signals