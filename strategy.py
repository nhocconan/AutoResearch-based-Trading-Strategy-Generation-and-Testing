#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + 1d EMA34 Trend + Volume Spike + Choppiness Filter
Hypothesis: On 12h timeframe, Donchian breakouts capture medium-term trends while avoiding noise.
EMA34 on 1d filters for higher timeframe trend alignment. Volume spike confirms conviction.
Choppiness index regime filter avoids whipsaw in sideways markets (CHOP > 61.8 = range, stay flat).
Designed for BTC/ETH with 50-150 total trades over 4 years to minimize fee drag.
Works in bull markets via breakout continuation and in bear markets via short breakdowns.
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
    
    # Get 1d data for EMA34 trend and choppiness filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need 34 for EMA34 + enough for chop calculation
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR(1)) / (n * log10(highest_high - lowest_low))) / log10(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = close_1d.values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d_arr[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d_arr[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First TR is undefined
    
    # ATR(1) = TR
    atr_1 = tr
    
    # Sum of ATR(1) over 14 periods
    sum_tr_14 = np.full(len(high_1d), np.nan)
    for i in range(14, len(high_1d)):
        sum_tr_14[i] = np.nansum(atr_1[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    max_high_14 = np.full(len(high_1d), np.nan)
    min_low_14 = np.full(len(low_1d), np.nan)
    for i in range(14, len(high_1d)):
        max_high_14[i] = np.max(high_1d[i-13:i+1])
        min_low_14[i] = np.min(low_1d[i-13:i+1])
    
    # Choppiness Index
    chop_1d = np.full(len(high_1d), np.nan)
    for i in range(14, len(high_1d)):
        if sum_tr_14[i] > 0 and (max_high_14[i] - min_low_14[i]) > 0:
            chop_1d[i] = 100 * np.log10(sum_tr_14[i] / (14 * np.log10(max_high_14[i] - min_low_14[i]))) / np.log10(14)
        else:
            chop_1d[i] = np.nan
    
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 20-period volume MA for volume spike confirmation (12h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA34, chop, volume MA, and Donchian
    start_idx = max(34, 20, 20)  # 34 for EMA34/chop, 20 for volume MA and Donchian
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_34_val = ema_34_1d_aligned[i]
        chop_val = chop_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Calculate Donchian channels (20-period)
        if i >= 20:
            donch_high = np.max(high[i-20:i])  # High of previous 20 bars
            donch_low = np.min(low[i-20:i])    # Low of previous 20 bars
        else:
            donch_high = np.nan
            donch_low = np.nan
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        # Choppiness regime filter: only trade when NOT choppy (CHOP <= 61.8 = trending)
        not_choppy = chop_val <= 61.8
        
        if position == 0:
            if not_choppy:
                # Look for breakouts in direction of 1d EMA34 trend
                if curr_close > donch_high and volume_confirm:
                    # Bullish breakout above upper Donchian
                    signals[i] = 0.25
                    position = 1
                elif curr_close < donch_low and volume_confirm:
                    # Bearish breakdown below lower Donchian
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
                    position = 0
            else:
                # Choppy market: stay flat
                signals[i] = 0.0
                position = 0
        elif position == 1:
            # Exit long: price closes below Donchian middle OR chop becomes extreme
            donch_mid = (donch_high + donch_low) / 2
            if curr_close < donch_mid or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian middle OR chop becomes extreme
            donch_mid = (donch_high + donch_low) / 2
            if curr_close > donch_mid or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0