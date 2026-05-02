#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and chop regime filter
# Donchian channel breakout captures institutional momentum moves
# 1d volume spike (2x 20-period EMA) confirms breakout validity
# Chopiness index > 61.8 defines ranging market to avoid false breakouts in consolidation
# Designed for 4h timeframe targeting 20-50 trades/year (80-200 total over 4 years)
# Uses discrete position sizing (0.25) to minimize fee churn and control drawdown
# Works in bull markets (breakout + volume + trending chop<61.8) and bear markets (breakdown + volume + trending chop<61.8)

name = "4h_Donchian20_1dVolume_Chop_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for volume confirmation and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for chop calculation
        return np.zeros(n)
    
    # 1d Chopiness Index calculation
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = pd.Series(df_1d['low']).diff().abs()
    tr3 = (pd.Series(df_1d['close']).shift() - pd.Series(df_1d['high'])).abs()
    tr4 = (pd.Series(df_1d['close']).shift() - pd.Series(df_1d['low'])).abs()
    tr = pd.concat([tr1, tr2, tr3, tr4], axis=1).max(axis=1)
    atr_sum = tr.rolling(window=14, min_periods=14).sum()
    
    # High-Low range
    hh = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max()
    ll = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min()
    hl_range = hh - ll
    
    # Chopiness Index: 100 * log10(atr_sum / hl_range) / log10(14)
    chop_raw = 100 * np.log10(atr_sum / hl_range) / np.log10(14)
    chop_values = chop_raw.fillna(50).values  # Neutral value when undefined
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # 1d Volume confirmation
    vol_ema_20 = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = (df_1d['volume'].values > (2.0 * vol_ema_20)).astype(float)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirmation)
    
    # 4h Donchian Channel (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = lookback
    
    for i in range(start_idx, n):
        if (np.isnan(chop_1d_aligned[i]) or np.isnan(volume_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Trending market filter: Chop < 61.8 (not ranging)
        trending_market = chop_1d_aligned[i] < 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above Donchian high with volume confirmation and trending market
            if close[i] > highest_high[i] and volume_1d_aligned[i] > 0.5 and trending_market:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below Donchian low with volume confirmation and trending market
            elif close[i] < lowest_low[i] and volume_1d_aligned[i] > 0.5 and trending_market:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Donchian low OR market becomes ranging
            if close[i] < lowest_low[i] or not trending_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian high OR market becomes ranging
            if close[i] > highest_high[i] or not trending_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals