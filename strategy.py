#!/usr/bin/env python3
"""
12h Williams Alligator + Volume Spike + Chop Regime Filter
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and strength.
In strong trends (Lips above Teeth above Jaw for long, reverse for short), price pulls back to
Teeth (8-period SMA) offer high-probability entries. Volume spike confirms momentum,
Chop regime filter avoids whipsaws in ranging markets. Designed for 12h timeframe
to target 12-37 trades/year by requiring Alligator alignment, volume confirmation,
and trending regime (Chop < 38.2). Works in bull/bear via trend-following logic.
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
    
    # Load 1d data ONCE before loop for Williams Alligator and Chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator on 1d: Jaw (13-period SMA, shifted 8), Teeth (8-period SMA, shifted 5), Lips (5-period SMA, shifted 3)
    # Using close prices
    close_1d = df_1d['close'].values
    jaw_1d = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth_1d = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    lips_1d = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to 12h
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Chop regime on 1d: Chop = 100 * log10(sum(ATR(1),14) / (log10(14) * (max(high,14) - min(low,14))))
    # Simplified: Chop > 61.8 = ranging, Chop < 38.2 = trending
    # We'll use a proxy: BB Width percentile (lower = trending)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bb_mid = pd.Series(close_1d).rolling(window=20, min_periods=20).mean()
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std()
    bb_width = (bb_mid + 2 * bb_std - (bb_mid - 2 * bb_std)) / bb_mid  # (upper - lower) / mid
    bb_width_percentile = bb_width.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    # Chop regime: trending when BB Width percentile < 40 (lower volatility = trending after expansion)
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    trending_regime = chop_regime_aligned < 40  # BB width in lower 40% = trending regime
    
    # Volume confirmation: current volume > 1.5 * 20-period average (moderate for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 13)  # BB width, Alligator
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or np.isnan(chop_regime_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        is_trending = trending_regime[i]
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
        bullish_alignment = (lips_1d_aligned[i] > teeth_1d_aligned[i] > jaw_1d_aligned[i])
        bearish_alignment = (lips_1d_aligned[i] < teeth_1d_aligned[i] < jaw_1d_aligned[i])
        
        if position == 0:
            # Look for entry signals - require ALL conditions: Alligator alignment + volume spike + trending regime
            # Long: bullish alignment AND price pulls back to Teeth (or slightly below) AND volume spike AND trending
            long_entry = bullish_alignment and (curr_low <= teeth_1d_aligned[i] * 1.002) and vol_spike and is_trending
            # Short: bearish alignment AND price pulls back to Teeth (or slightly above) AND volume spike AND trending
            short_entry = bearish_alignment and (curr_high >= teeth_1d_aligned[i] * 0.998) and vol_spike and is_trending
            
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
            # Exit: price closes below Jaw (trend reversal) OR loss of bullish alignment
            if (curr_close < jaw_1d_aligned[i]) or not bullish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price closes above Jaw (trend reversal) OR loss of bearish alignment
            if (curr_close > jaw_1d_aligned[i]) or not bearish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_VolumeSpike_TrendingRegime"
timeframe = "12h"
leverage = 1.0