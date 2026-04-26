#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hTrend_VolumeSpike_ChopRegime_v1
Hypothesis: 4h Donchian(20) breakout with 12h EMA(50) trend filter, volume spike confirmation, and choppiness regime filter (CHOP > 61.8 for mean reversion, < 38.2 for trend). 
Long when price breaks above upper Donchian AND volume spike AND 12h uptrend AND chop < 38.2 (trending). 
Short when price breaks below lower Donchian AND volume spike AND 12h downtrend AND chop < 38.2 (trending).
Exit when price fails to hold breakout level or regime shifts to chop (CHOP > 61.8).
Uses discrete position size 0.25 to minimize fee churn. Targets 20-50 trades/year.
Works in bull via breakout continuation and bear via breakdown continuation with 12h trend filter.
"""

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
    
    # Get 12h data for trend filter and chop regime
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Choppiness Index on 12h (regime filter)
    # CHOP = 100 * log10(sum(ATR(14)) / log10(range)) / log10(N)
    # where ATR = TR, range = HHV - LLV over N periods
    # Simplified: CHOP = 100 * log10(sum(TR) / (HHV - LLV)) / log10(N)
    # We'll use 14-period CHOP as standard
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h_arr = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h_arr[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h_arr[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first TR is undefined
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # HHV and LLV over 14 periods
    hh = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    range_hl = hh - ll
    
    # Choppiness Index
    chop = 100 * np.log10(tr_sum / range_hl) / np.log10(14)
    chop = np.where(range_hl == 0, 100, chop)  # avoid division by zero
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # Calculate Donchian(20) on 4h (primary timeframe)
    # Upper = highest high over 20 periods, Lower = lowest low over 20 periods
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of 12h EMA(50), Donchian(20), volume MA(20), chop calculation
    start_idx = max(50, 20, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(chop_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        vol_conf = volume_confirm[i]
        trend_up = close_val > ema_50_12h_aligned[i]   # 12h uptrend
        trend_down = close_val < ema_50_12h_aligned[i]  # 12h downtrend
        chop_val = chop_aligned[i]
        
        # Regime filter: only trade when market is trending (CHOP < 38.2)
        is_trending = chop_val < 38.2
        
        if position == 0:
            # Long: price breaks above upper Donchian AND volume spike AND 12h uptrend AND trending regime
            long_signal = (close_val > highest_high[i]) and vol_conf and trend_up and is_trending
            
            # Short: price breaks below lower Donchian AND volume spike AND 12h downtrend AND trending regime
            short_signal = (close_val < lowest_low[i]) and vol_conf and trend_down and is_trending
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price drops below upper Donchian (failed breakout) OR regime shifts to chop OR 12h trend flips down
            if (close_val < highest_high[i]) or (not is_trending) or (not trend_up):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above lower Donchian (failed breakdown) OR regime shifts to chop OR 12h trend flips up
            if (close_val > lowest_low[i]) or (not is_trending) or (not trend_down):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_12hTrend_VolumeSpike_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0