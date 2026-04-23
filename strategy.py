#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation + chop regime filter.
Long when price breaks above Donchian upper band AND 1d EMA34 is rising AND volume > 1.5x 20-period average AND chop < 61.8.
Short when price breaks below Donchian lower band AND 1d EMA34 is falling AND volume > 1.5x 20-period average AND chop < 61.8.
Exit when price touches the opposite Donchian band or chop > 61.8 (range regime).
Uses 1d HTF for EMA34 trend to reduce whipsaws and chop filter to avoid ranging markets.
Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d chop regime filter (HTF)
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chopiness Index: 100 * log10(sum(ATR14)/ (HH14-LL14)) / log10(14)
    chop = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        if atr_14[i] <= 0 or hh_14[i] == ll_14[i]:
            continue
        sum_atr = np.sum(atr_14[i-13:i+1])  # sum of last 14 ATR values
        chop[i] = 100 * np.log10(sum_atr / (hh_14[i] - ll_14[i])) / np.log10(14)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 4h Donchian channels (20-period)
    donch_h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_l = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 (34), Donchian (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(donch_h[i]) or np.isnan(donch_l[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_34_aligned[i]
        chop_val = chop_aligned[i]
        upper = donch_h[i]
        lower = donch_l[i]
        vol_ma_val = vol_ma[i]
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            # Long: Break above Donchian upper AND EMA34 rising AND volume spike AND chop < 61.8 (trending)
            if price > upper and ema_rising and volume[i] > 1.5 * vol_ma_val and chop_val < 61.8:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower AND EMA34 falling AND volume spike AND chop < 61.8 (trending)
            elif price < lower and ema_falling and volume[i] > 1.5 * vol_ma_val and chop_val < 61.8:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches lower band OR chop > 61.8 (range regime)
                if price < lower or chop_val > 61.8:
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches upper band OR chop > 61.8 (range regime)
                if price > upper or chop_val > 61.8:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_1dEMA34_Trend_Volume_ChopFilter"
timeframe = "4h"
leverage = 1.0