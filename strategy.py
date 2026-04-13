#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + 1d volume confirmation + chop regime filter
    # Long: price > upper Donchian(20) AND volume > 1.5x 20-period avg AND chop < 61.8 (trending)
    # Short: price < lower Donchian(20) AND volume > 1.5x 20-period avg AND chop < 61.8 (trending)
    # Exit: price crosses opposite Donchian band OR chop > 61.8 (range) OR volume dry-up
    # Using 4h timeframe for optimal trade frequency, Donchian for structure,
    # volume for confirmation, chop regime to avoid false breakouts in ranging markets.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily chop regime (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR(14)
    atr = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        if i == 14:
            atr[i] = np.nanmean(tr[1:15])  # first 14 TR values
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # highest high and lowest low over 14 periods
    hh = np.full(len(close_1d), np.nan)
    ll = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        hh[i] = np.max(high_1d[i-13:i+1])
        ll[i] = np.min(low_1d[i-13:i+1])
    
    # Chop = 100 * log10(sum(ATR)/log10(hh-ll)) / log10(14)
    chop = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        if hh[i] > ll[i] and not np.isnan(atr[i]):
            sum_atr = np.sum(atr[i-13:i+1])
            chop[i] = 100 * np.log10(sum_atr) / np.log10(14) / np.log10(hh[i] - ll[i])
        else:
            chop[i] = np.nan
    
    # Align chop to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 4h Donchian channels (20-period)
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(lookback-1, n):
        upper[i] = np.max(high[i-lookback+1:i+1])
        lower[i] = np.min(low[i-lookback+1:i+1])
    
    # Get 4h volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: chop < 61.8 = trending (favor breakouts)
        trending_regime = chop_aligned[i] < 61.8
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry logic: Donchian breakout + trending regime + volume confirmation
        long_entry = (close[i] > upper[i]) and trending_regime and vol_confirm
        short_entry = (close[i] < lower[i]) and trending_regime and vol_confirm
        
        # Exit logic: opposite Donchian touch OR chop > 61.8 (range) OR volume dry-up
        long_exit = (close[i] < lower[i]) or (chop_aligned[i] >= 61.8) or not vol_confirm
        short_exit = (close[i] > upper[i]) or (chop_aligned[i] >= 61.8) or not vol_confirm
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_breakout_volume_chop_v2"
timeframe = "4h"
leverage = 1.0