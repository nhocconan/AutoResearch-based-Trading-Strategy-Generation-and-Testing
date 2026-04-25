#!/usr/bin/env python3
"""
4h Donchian Breakout with 1d EMA Trend and Volume Spike + Chop Filter
Hypothesis: Donchian(20) breakouts capture institutional momentum. 1d EMA34 provides multi-day trend filter,
volume spike confirms participation, and choppiness index avoids false signals in ranging markets.
Works in bull markets (long breakouts above upper band in uptrend) and bear markets (short breakouts below lower band in downtrend).
Target: 20-50 trades/year per symbol. Discrete sizing: 0.25.
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Choppiness Index: chop > 61.8 = ranging (avoid entries), chop < 38.2 = trending (allow entries)
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Max(high) - Min(low) over 14 periods
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    range_max_min = max_high - min_low
    chop = np.where(range_max_min != 0, 100 * np.log10(tr_sum / range_max_min) / np.log10(14), 50)
    
    # Donchian channels (20-period)
    upper_channel = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all calculations
    start_idx = max(20, 34, 14)  # Donchian, EMA, ATR/CHOP
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        vol_spike = volume_spike[i]
        curr_chop = chop[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_aligned[i]
        downtrend = curr_close < ema_34_aligned[i]
        
        # Only allow entries in trending markets (chop < 38.2)
        trending_market = curr_chop < 38.2
        
        if position == 0:
            # Look for entry signals
            # Long: break above upper Donchian channel AND uptrend AND volume spike AND trending market
            long_entry = (curr_close > upper_channel[i]) and uptrend and vol_spike and trending_market
            # Short: break below lower Donchian channel AND downtrend AND volume spike AND trending market
            short_entry = (curr_close < lower_channel[i]) and downtrend and vol_spike and trending_market
            
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
            # Exit: price breaks below lower channel (reversal) OR loss of uptrend OR chop too high (ranging)
            if (curr_close < lower_channel[i]) or (curr_close < ema_34_aligned[i]) or (curr_chop > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price breaks above upper channel (reversal) OR loss of downtrend OR chop too high (ranging)
            if (curr_close > upper_channel[i]) or (curr_close > ema_34_aligned[i]) or (curr_chop > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_1dEMA34_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0