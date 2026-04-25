#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Donchian channel breakouts capture strong momentum moves. 
Filtering by 1d EMA34 trend ensures alignment with higher timeframe direction.
Volume spike confirms institutional participation. 
Choppiness filter avoids whipsaws in ranging markets.
Designed for 4h timeframe with tight entry conditions targeting 20-50 trades/year.
Works in bull (breakouts above upper channel in uptrend) and bear 
(breakouts below lower channel in downtrend).
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
    
    # Get 1d data for EMA and chop filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Choppiness Index on 1d (14-period)
    # CHOP = 100 * log10(sum(ATR(1),14) / (log10(HH(14)-LL(14)) * sqrt(14)))
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with 1d index
    
    atr_1 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_hl = highest_high - lowest_low
    chop = np.where(range_hl > 0, 
                    100 * np.log10(pd.Series(atr_1).rolling(window=14, min_periods=14).sum().values / 
                                   (np.log10(range_hl) * np.sqrt(14))), 
                    50)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Donchian(20) on 4h
    donchian_window = 20
    highest_high_4h = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lowest_low_4h = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian, EMA, and volume MA
    start_idx = max(20, 34, 20) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high_4h[i]) or np.isnan(lowest_low_4h[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        chop_value = chop_aligned[i]
        vol_spike = volume_spike[i]
        upper_channel = highest_high_4h[i]
        lower_channel = lowest_low_4h[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above upper Donchian channel AND volume spike AND 
            #       price > EMA (uptrend) AND market is trending (CHOP < 38.2)
            long_entry = (curr_high > upper_channel) and vol_spike and (curr_close > ema_trend) and (chop_value < 38.2)
            # Short: price breaks below lower Donchian channel AND volume spike AND 
            #        price < EMA (downtrend) AND market is trending (CHOP < 38.2)
            short_entry = (curr_low < lower_channel) and vol_spike and (curr_close < ema_trend) and (chop_value < 38.2)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below lower Donchian channel OR price crosses below EMA (trend change) OR chop becomes too high
            if (curr_low < lower_channel) or (curr_close < ema_trend) or (chop_value > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above upper Donchian channel OR price crosses above EMA (trend change) OR chop becomes too high
            if (curr_high > upper_channel) or (curr_close > ema_trend) or (chop_value > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0