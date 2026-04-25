#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Donchian breakouts on 12h capture medium-term momentum. 1d EMA34 filters for higher-timeframe trend alignment. Volume spike confirms breakout strength. Choppiness index filter (CHOP > 61.8) avoids false breakouts in ranging markets. Designed for 12h timeframe to target 12-37 trades/year (50-150 over 4 years), minimizing fee drag. Works in both bull and bear markets by following the 1d trend and avoiding counter-trend entries.
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
    
    # Align Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # 1d EMA34 for trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average (stricter for 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Choppiness index filter: avoid ranging markets
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (max(high) - min(low)))) 
    # Simplified: CHOP > 61.8 = ranging (avoid), CHOP < 38.2 = trending (favor)
    tr_range = np.maximum(high - low, 
                         np.maximum(np.abs(high - np.roll(close, 1)), 
                                   np.abs(low - np.roll(close, 1))))
    # Handle first bar
    tr_range[0] = high[0] - low[0]
    atr14 = pd.Series(tr_range).rolling(window=14, min_periods=14).mean().values
    max_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    min_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    chop = 100 * np.log10(atr14 * 14 / np.log10(14) / (max_high_20 - min_low_20 + 1e-10))
    chop_filter = chop > 61.8  # True when ranging (avoid entries)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 34)  # Donchian, EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        is_ranging = chop_filter[i]
        
        # Trend filter: price relative to 1d EMA34
        bullish_bias = curr_close > ema_1d_aligned[i]
        bearish_bias = curr_close < ema_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals - ONLY in trending markets (not ranging)
            if not is_ranging:
                # Long: price breaks above Donchian high AND bullish bias AND volume spike
                long_entry = (curr_high > donchian_high_aligned[i]) and bullish_bias and vol_spike
                # Short: price breaks below Donchian low AND bearish bias AND volume spike
                short_entry = (curr_low < donchian_low_aligned[i]) and bearish_bias and vol_spike
                
                if long_entry:
                    signals[i] = 0.25
                    position = 1
                elif short_entry:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # No entries in ranging markets
        elif position == 1:
            # Long position management
            # Exit: price falls below Donchian low (breakdown) OR loss of bullish bias
            if (curr_low < donchian_low_aligned[i]) or (curr_close < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above Donchian high (breakout) OR loss of bearish bias
            if (curr_high > donchian_high_aligned[i]) or (curr_close > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0