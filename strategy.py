#!/usr/bin/env python3
"""
4h Volume Spike Breakout + 1d EMA Trend + Choppiness Filter
Hypothesis: Volume spikes confirm institutional participation in breakouts.
In trending regimes (Choppiness Index < 38.2), breakouts in the trend direction
have higher probability of continuation. In ranging regimes (Choppiness Index > 61.8),
we avoid breakout trades to prevent whipsaws. Uses 4h timeframe targeting 20-50 trades/year.
Works in both bull and bear markets by filtering trades to trend regime only.
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
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian channels (20-period) on 4h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Choppiness Index on 1d (to filter regime)
    # CHOP = 100 * LOG10(SUM(ATR(1), n) / (LOG10(MAX(HIGH,n) - MIN(LOW,n)))) / LOG10(n)
    # We calculate a simplified version: ATR(14) / (HHV(14) - LLV(14)) * 100
    tr1 = np.maximum(df_1d['high'].values, np.roll(df_1d['close'].values, 1)) - np.minimum(df_1d['low'].values, np.roll(df_1d['close'].values, 1))
    tr1[0] = df_1d['high'].values[0] - df_1d['low'].values[0]  # first TR
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    hh14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop_denom = hh14 - ll14
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid division by zero
    chop_raw = (atr1 / chop_denom) * 100
    # Smooth chop with MA
    chop = pd.Series(chop_raw).rolling(window=14, min_periods=14).mean().values
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 34, 14)  # Donchian, EMA34, chop
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        chop_val = chop_aligned[i]
        
        # Regime filter: only trade in trending markets (CHOP < 38.2)
        trending_regime = chop_val < 38.2
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Donchian high AND volume spike AND trending regime AND uptrend (price > 1d EMA34)
            long_entry = (curr_high > high_20[i]) and vol_spike and trending_regime and (curr_close > ema_34_aligned[i])
            # Short: price breaks below Donchian low AND volume spike AND trending regime AND downtrend (price < 1d EMA34)
            short_entry = (curr_low < low_20[i]) and vol_spike and trending_regime and (curr_close < ema_34_aligned[i])
            
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
            # Exit: price breaks below Donchian low OR loss of volume momentum OR regime change to ranging
            if (curr_low < low_20[i]) or (not vol_spike) or (chop_val > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price breaks above Donchian high OR loss of volume momentum OR regime change to ranging
            if (curr_high > high_20[i]) or (not vol_spike) or (chop_val > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_VolumeSpike_Breakout_1dEMA_Trend_ChopFilter"
timeframe = "4h"
leverage = 1.0