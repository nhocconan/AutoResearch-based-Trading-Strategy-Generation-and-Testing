#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + Volume Spike + Chop Regime Filter
Hypothesis: Donchian breakouts capture momentum bursts. Volume confirmation ensures institutional participation. Choppiness index filter avoids false signals in ranging markets. Works in bull markets (trend continuation) and bear markets (mean reversion at extremes) by using regime-aware exits.
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
    
    # Get 1d data for Chop regime (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        tr1[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
    
    # Sum of TR over 14 periods
    tr_sum = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        tr_sum[i] = np.sum(tr1[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    hh = np.full(len(close_1d), np.nan)
    ll = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        hh[i] = np.max(high_1d[i-13:i+1])
        ll[i] = np.min(low_1d[i-13:i+1])
    
    # Chop = 100 * log10(sum(tr1)/ (hh - ll)) / log10(14)
    chop = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        if hh[i] > ll[i]:
            chop[i] = 100 * np.log10(tr_sum[i] / (hh[i] - ll[i])) / np.log10(14)
    
    # Align CHOP to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate ATR(14) for dynamic stop and position sizing
    atr = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for ATR, volume MA, and Donchian
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        chop_val = chop_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        
        # Calculate Donchian channels (20-period)
        if i >= 20:
            donch_high = np.max(high[i-19:i+1])
            donch_low = np.min(low[i-19:i+1])
        else:
            donch_high = np.nan
            donch_low = np.nan
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        # Chop regime: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow)
        # We use CHOP > 50 as a softer filter to avoid whipsaw in strong trends
        chop_filter = chop_val > 50  # Avoid trading in very choppy markets
        
        if position == 0:
            # Long: price breaks above Donchian high with volume confirmation and not too choppy
            long_breakout = (curr_close > donch_high) and volume_confirm and not chop_filter
            # Short: price breaks below Donchian low with volume confirmation and not too choppy
            short_breakout = (curr_close < donch_low) and volume_confirm and not chop_filter
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price closes below Donchian low OR 2*ATR stoploss
            if curr_close < donch_low or curr_close < (entry_price - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price closes above Donchian high OR 2*ATR stoploss
            if curr_close > donch_high or curr_close > (entry_price + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_VolumeSpike_ChopFilter_ATRStop"
timeframe = "4h"
leverage = 1.0