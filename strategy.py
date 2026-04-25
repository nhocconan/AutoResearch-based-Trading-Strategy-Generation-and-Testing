#!/usr/bin/env python3
"""
12h Williams Alligator + 1d EMA50 Trend + Volume Spike
Hypothesis: Williams Alligator identifies trending vs ranging markets via jaw/teeth/lips alignment. 
In trending markets (Alligator awake), we follow 1d EMA50 direction with Donchian(20) breakouts for entry.
Volume spike confirms conviction. Chop filter avoids false signals. Designed for 12h timeframe to target 12-37 trades/year.
Works in bull markets by catching trends, in bear markets by avoiding counter-trend trades and catching short opportunities.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_alligator

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
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Williams Alligator on 1d: Jaw(13,8), Teeth(8,5), Lips(5,3)
    jaw, teeth, lips = compute_williams_alligator(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    
    # Align Alligator lines to 12h timeframe (no extra delay needed for SMMA)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # 1d EMA50 for trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Donchian channels (20-period) on 1d for breakout entries
    donchian_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Chop filter: avoid trading in choppy markets (Choppiness Index > 61.8)
    tr1 = pd.Series(high).rolling(2).max() - pd.Series(low).rolling(2).min()
    tr2 = abs(pd.Series(high) - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low) - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr / np.maximum(hh - ll, 1e-10)) / np.log10(14)
    chop_filter = chop < 61.8  # Only trade when NOT choppy (trending market)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(50, 20, 20, 14, 14)  # EMA, Donchian, volume MA, Chop components
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        chop_ok = chop_filter[i]
        
        # Alligator: market is trending when lips > teeth > jaw (bullish) or lips < teeth < jaw (bearish)
        bullish_alligator = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
        bearish_alligator = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
        
        # Trend filter: price relative to 1d EMA50
        bullish_bias = curr_close > ema_1d_aligned[i]
        bearish_bias = curr_close < ema_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals - require ALL conditions: Alligator alignment + trend + breakout + volume + chop filter
            # Long: bullish Alligator AND bullish bias AND price breaks above Donchian high AND volume spike AND chop filter
            long_entry = bullish_alligator and bullish_bias and (curr_high > donchian_high_aligned[i]) and vol_spike and chop_ok
            # Short: bearish Alligator AND bearish bias AND price breaks below Donchian low AND volume spike AND chop filter
            short_entry = bearish_alligator and bearish_bias and (curr_low < donchian_low_aligned[i]) and vol_spike and chop_ok
            
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
            # Exit: Alligator turns bearish OR price breaks below Donchian low OR loss of bullish bias OR chop becomes too high
            if (not bullish_alligator) or (curr_low < donchian_low_aligned[i]) or (curr_close < ema_1d_aligned[i]) or (not chop_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Alligator turns bullish OR price breaks above Donchian high OR loss of bearish bias OR chop becomes too high
            if (not bearish_alligator) or (curr_high > donchian_high_aligned[i]) or (curr_close > ema_1d_aligned[i]) or (not chop_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0