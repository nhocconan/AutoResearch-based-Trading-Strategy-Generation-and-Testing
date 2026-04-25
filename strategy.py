#!/usr/bin/env python3
"""
12h Williams Alligator + Chop Regime + Volume Spike
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trends; Chop regime filters false signals; Volume spike confirms momentum.
Works in bull (Alligator long alignment + chop exit) and bear (short alignment + chop exit) with 12h/1d timeframe to limit trades.
Target: 12-30 trades/year by requiring confluence of Alligator alignment, chop regime, and volume spike.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator on 1d: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs of median price
    median_price = (df_1d['high'] + df_1d['low']) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Chop regime on 1d: Chop = 100 * log10(sum(ATR(1),14) / (log10(14) * (HHV(high,14) - LLV(low,14))))
    # Simplified: Chop > 61.8 = ranging, Chop < 38.2 = trending
    atr_1d = pd.Series(np.abs(df_1d['high'] - df_1d['low'])).rolling(window=14, min_periods=14).mean().values
    sum_atr = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr / (np.log10(14) * (hh - ll))) if np.all(hh - ll > 0) else np.full_like(sum_atr, 50.0)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume spike on primary: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator and Chop
    start_idx = 50  # Alligator lookback
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish; Lips < Teeth < Jaw = bearish
        bullish_alignment = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
        bearish_alignment = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
        
        # Chop regime: trending when Chop < 38.2 (strong trend), ranging when Chop > 61.8
        trending_regime = chop_aligned[i] < 38.2
        ranging_regime = chop_aligned[i] > 61.8
        
        if position == 0:
            # Look for entry signals - require ALL conditions: Alligator alignment + trending regime + volume spike
            # Long: bullish alignment AND trending regime AND volume spike
            long_entry = bullish_alignment and trending_regime and vol_spike
            # Short: bearish alignment AND trending regime AND volume spike
            short_entry = bearish_alignment and trending_regime and vol_spike
            
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
            # Exit: loss of bullish alignment OR chop becomes ranging (Chop > 61.8) OR volume dries up
            if (not bullish_alignment) or ranging_regime or (not vol_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: loss of bearish alignment OR chop becomes ranging OR volume dries up
            if (not bearish_alignment) or ranging_regime or (not vol_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_ChopRegime_VolumeSpike"
timeframe = "12h"
leverage = 1.0