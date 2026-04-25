#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Donchian channel breakouts on 12h capture medium-term momentum. EMA34 on 1d ensures trend alignment, volume spike confirms conviction, and choppiness filter avoids range-bound whipsaws. Designed for 12h timeframe to target 12-37 trades/year (50-150 over 4 years), minimizing fee drag. Works in both bull and bear markets by following the 1d trend and avoiding counter-trend entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 1d
    donchian_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align Donchian levels to 12h timeframe
    dh_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    dl_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    dm_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    
    # 1d EMA34 for trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Choppiness filter: avoid trading in choppy markets (CHOP > 61.8)
    # True range = max(high-low, abs(high-prev_close), abs(low-prev_close))
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) = average true range over 14 periods
    atr = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    
    # Sum of true range over 14 periods
    sum_tr = pd.Series(true_range).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index formula: 100 * log10(sum_tr / (hh - ll)) / log10(14)
    # Avoid division by zero
    hh_minus_ll = hh - ll
    chop = np.where(hh_minus_ll > 0, 100 * np.log10(sum_tr / hh_minus_ll) / np.log10(14), 50)
    chop_filter = chop < 61.8  # Only trade when NOT choppy (trending market)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 34, 20, 14)  # Donchian, EMA, volume MA, chop
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(dh_aligned[i]) or np.isnan(dl_aligned[i]) or 
            np.isnan(dm_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        chop_ok = chop_filter[i]
        
        # Trend filter: price relative to 1d EMA34
        bullish_bias = curr_close > ema_1d_aligned[i]
        bearish_bias = curr_close < ema_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Donchian high AND bullish bias AND volume spike AND not choppy
            long_entry = (curr_high > dh_aligned[i]) and bullish_bias and vol_spike and chop_ok
            # Short: price breaks below Donchian low AND bearish bias AND volume spike AND not choppy
            short_entry = (curr_low < dl_aligned[i]) and bearish_bias and vol_spike and chop_ok
            
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
            # Exit: price falls below Donchian mid (mean reversion) OR loss of bullish bias
            if (curr_low < dm_aligned[i]) or (curr_close < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above Donchian mid (mean reversion) OR loss of bearish bias
            if (curr_high > dm_aligned[i]) or (curr_close > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0