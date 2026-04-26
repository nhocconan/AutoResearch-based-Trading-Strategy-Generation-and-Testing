#!/usr/bin/env python3
"""
6h_IBS_MeanReversion_WeeklyTrendFilter_VolumeSpike_v1
Hypothesis: On 6h timeframe, use Internal Bar Strength (IBS) for mean reversion entries, 
filtered by weekly trend direction (price above/below weekly EMA20) and volume spike confirmation. 
IBS = (close - low) / (high - low). Long when IBS < 0.2 (oversold) in weekly uptrend, 
short when IBS > 0.8 (overbought) in weekly downtrend. Volume spike confirms institutional participation. 
Designed for 12-30 trades/year to minimize fee drag while working in both bull and bear markets 
by taking mean reversion trades only with the weekly trend.
"""

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
    
    # Calculate weekly EMA20 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 6h IBS (Internal Bar Strength)
    # IBS = (close - low) / (high - low), ranges from 0 to 1
    hl_range = high - low
    # Avoid division by zero
    ibs = np.where(hl_range != 0, (close - low) / hl_range, 0.5)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # ATR for stoploss (14-period on 6h)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for all indicators
    start_idx = max(20, 20, 14)  # weekly EMA, volume avg, ATR
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(atr[i]) or np.isnan(ibs[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        atr_val = atr[i]
        weekly_trend = ema_1w_aligned[i]
        ibs_val = ibs[i]
        size = 0.25  # 25% position size to manage risk
        
        if position == 0:
            # Flat - look for mean reversion in direction of weekly trend with volume confirmation
            # Long: weekly uptrend (price > weekly EMA20) AND IBS < 0.2 (oversold) + volume spike
            long_entry = (close_val > weekly_trend) and (ibs_val < 0.2) and volume_spike[i]
            # Short: weekly downtrend (price < weekly EMA20) AND IBS > 0.8 (overbought) + volume spike
            short_entry = (close_val < weekly_trend) and (ibs_val > 0.8) and volume_spike[i]
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on IBS mean reversion (IBS > 0.5) or ATR stoploss
            exit_condition = (ibs_val > 0.5) or \
                           (close_val < entry_price - 2.5 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on IBS mean reversion (IBS < 0.5) or ATR stoploss
            exit_condition = (ibs_val < 0.5) or \
                           (close_val > entry_price + 2.5 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_IBS_MeanReversion_WeeklyTrendFilter_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0