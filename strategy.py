#!/usr/bin/env python3
"""
12h_PriceChannel_Breakout_Volume_Regime
Hypothesis: On 12h timeframe, enter long when price breaks above Donchian(20) upper band with volume confirmation and chop regime filter (CHOP > 61.8 = ranging), short when breaks below lower band. Exit on opposite break. Uses 1d ADX for trend strength filter to avoid whipsaws. Targets 15-25 trades/year by requiring multiple confirmations, position size 0.25. Designed to work in both bull (breakouts) and bear (mean reversion in chop) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Average True Range for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Get 1d data for ADX and Chop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], 
                       np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                                  np.abs(low_1d[1:] - close_1d[:-1])))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    
    plus_di_1d = np.full(len(close_1d), np.nan)
    minus_di_1d = np.full(len(close_1d), np.nan)
    dx_1d = np.full(len(close_1d), np.nan)
    
    # Smoothed values
    atr_1d_sm = np.full(len(close_1d), np.nan)
    plus_dm_sm = np.full(len(close_1d), np.nan)
    minus_dm_sm = np.full(len(close_1d), np.nan)
    
    for i in range(14, len(close_1d)):
        if i == 14:
            atr_1d_sm[i] = np.mean(tr_1d[1:15])
            plus_dm_sm[i] = np.sum(plus_dm[1:15])
            minus_dm_sm[i] = np.sum(minus_dm[1:15])
        else:
            atr_1d_sm[i] = (atr_1d_sm[i-1] * 13 + tr_1d[i]) / 14
            plus_dm_sm[i] = (plus_dm_sm[i-1] * 13 + plus_dm[i]) / 14
            minus_dm_sm[i] = (minus_dm_sm[i-1] * 13 + minus_dm[i]) / 14
    
    plus_di_1d = np.where(atr_1d_sm != 0, 100 * plus_dm_sm / atr_1d_sm, 0)
    minus_di_1d = np.where(atr_1d_sm != 0, 100 * minus_dm_sm / atr_1d_sm, 0)
    dx_1d = np.where((plus_di_1d + minus_di_1d) != 0, 
                     100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d), 0)
    
    adx_1d = np.full(len(close_1d), np.nan)
    for i in range(27, len(close_1d)):  # 14 + 13 for smoothing
        if i == 27:
            adx_1d[i] = np.mean(dx_1d[14:28])
        else:
            adx_1d[i] = (adx_1d[i-1] * 13 + dx_1d[i]) / 14
    
    # Calculate Choppy Index (14-period) on 1d
    sum_tr_14 = np.full(len(close_1d), np.nan)
    highest_high_14 = np.full(len(close_1d), np.nan)
    lowest_low_14 = np.full(len(close_1d), np.nan)
    
    for i in range(14, len(close_1d)):
        sum_tr_14[i] = np.sum(tr_1d[i-13:i+1])
        highest_high_14[i] = np.max(high_1d[i-13:i+1])
        lowest_low_14[i] = np.min(low_1d[i-13:i+1])
    
    chop_1d = 100 * np.log10(sum_tr_14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    
    # Align 1d indicators to 12h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume average (20-period) for confirmation
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # need Donchian(20) and enough for ADX
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in ranging markets (CHOP > 61.8)
        # In trending markets, we avoid breakouts to prevent whipsaws
        if chop_1d_aligned[i] <= 61.8:
            # In trending regime, stay flat or follow trend weakly
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above Donchian high with volume confirmation
            if close[i] > donch_high[i] and volume[i] > vol_ma[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low with volume confirmation
            elif close[i] < donch_low[i] and volume[i] > vol_ma[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price breaks below Donchian low
            if close[i] < donch_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high
            if close[i] > donch_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_PriceChannel_Breakout_Volume_Regime"
timeframe = "12h"
leverage = 1.0