#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_VolumeChop_v1
Hypothesis: On 1d timeframe, Donchian(20) breakouts with 1w EMA50 trend filter, volume confirmation (>1.5x 20-day average), and choppiness regime filter (CHOP < 50) produce high-quality trades with low frequency. The 1w trend filter ensures alignment with the primary weekly trend, reducing counter-trend whipsaws. Volume confirmation adds conviction, and the chop filter avoids false breakouts in ranging markets. Designed to work in both bull and bear markets by following the 1w trend. Target: 30-100 total trades over 4 years (7-25/year).
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
    
    # Load 1w data ONCE before loop for HTF trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian(20) channels on 1d
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d volume confirmation: volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d choppiness index: CHOP > 61.8 = ranging (avoid breakouts), CHOP < 38.2 = trending (favor breakouts)
    # CHOP = 100 * log10(sum(ATR14) / (max(high,n) - min(low,n))) / log10(n)
    tr1 = np.maximum(high - low, np.absolute(high - np.concatenate([[np.nan], close[:-1]])))
    tr2 = np.maximum(tr1, np.absolute(low - np.concatenate([[np.nan], close[:-1]])))
    atr14 = pd.Series(tr2).rolling(window=14, min_periods=14).mean().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(pd.Series(atr14).rolling(window=14, min_periods=14).sum().values / (max_high_14 - min_low_14)) / np.log10(14)
    chop = np.where((max_high_14 - min_low_14) == 0, 50, chop)  # avoid div by zero
    chop = np.nan_to_num(chop, nan=50.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for EMA, 20 for Donchian/volume MA, 14*2 for chop)
    start_idx = max(50, 20, 28)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1w trend filter (EMA50)
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Choppiness regime: only take breakouts when CHOP < 50 (less choppy/more trending)
        regime_ok = chop[i] < 50.0
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high_20[i]
        breakout_down = close[i] < lowest_low_20[i]
        
        # Long logic: breakout above Donchian upper in uptrend with volume and good regime
        if uptrend and volume_spike and breakout_up and regime_ok:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: breakout below Donchian lower in downtrend with volume and good regime
        elif downtrend and volume_spike and breakout_down and regime_ok:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: loss of trend OR regime becomes too choppy
        elif position == 1 and (not uptrend or chop[i] >= 61.8):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not downtrend or chop[i] >= 61.8):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_VolumeChop_v1"
timeframe = "1d"
leverage = 1.0