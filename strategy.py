#!/usr/bin/env python3
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
    
    # Get 1d data for Camarilla pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla pivot levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    H4 = close_1d + (range_hl * 1.1 / 2)
    L4 = close_1d - (range_hl * 1.1 / 2)
    H3 = close_1d + (range_hl * 1.1 / 4)
    L3 = close_1d - (range_hl * 1.1 / 4)
    
    # Align pivot levels to 1d (primary timeframe)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Get 1w data for regime filter (choppiness index)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range for 1w
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1w = np.concatenate([[np.nan], tr_1w])
    
    # ATR(14) for 1w
    atr_1w = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods for 1w
    hh_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum(atr_1w,14) / (hh_1w - ll_1w)) / log10(14)
    atr_sum_1w = pd.Series(atr_1w).rolling(window=14, min_periods=14).sum().values
    chop_denom = hh_1w - ll_1w
    chop_denom = np.where(chop_denom == 0, np.nan, chop_denom)
    chop_raw = 100 * np.log10(atr_sum_1w / chop_denom) / np.log10(14)
    chop_1w = pd.Series(chop_raw).fillna(50).values  # fill NaN with neutral 50
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Volume confirmation from 1d
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(H4_aligned[i]) or 
            np.isnan(L4_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(chop_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from 1d EMA
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter: current 1d volume above average
        volume_filter = volume_1d[i] > vol_ma_20_1d_aligned[i]
        
        # Regime filter: avoid choppy markets (chop > 61.8 = range, chop < 38.2 = trending)
        # We want trending markets for breakout strategy
        regime_filter = chop_1w_aligned[i] < 61.8  # Allow trending and mildly choppy
        
        # Entry conditions: Camarilla H4/L4 breakout with volume and trend
        long_breakout = close[i] > H4_aligned[i]
        short_breakout = close[i] < L4_aligned[i]
        
        long_entry = uptrend and long_breakout and volume_filter and regime_filter
        short_entry = downtrend and short_breakout and volume_filter and regime_filter
        
        # Exit conditions: Camarilla H3/L3 retracement
        long_exit = close[i] < H3_aligned[i]
        short_exit = close[i] > L3_aligned[i]
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_Camarilla_H4L4_Breakout_VolumeTrend_Regime"
timeframe = "1d"
leverage = 1.0