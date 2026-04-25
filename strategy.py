#!/usr/bin/env python3
"""
12h Williams Alligator Breakout + 1d EMA50 Trend + Volume Spike + Chop Filter
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trend absence; breakout from Alligator's "sleep" (all lines intertwined) with volume confirmation captures strong moves. 1d EMA50 filters counter-trend trades. Chop filter (BBW percentile) avoids low-volatility false breakouts. Discrete sizing 0.25 limits drawdown. Designed for low trade frequency (<40/year) to overcome fee drag in bear markets like 2025.
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
    
    # Load 1d data ONCE before loop for EMA50 trend filter and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1d Bollinger Bands for chop regime (20, 2)
    bb_ma = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    bb_width = bb_upper - bb_lower
    bb_width_ma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    bb_width_percentile = bb_width / (bb_width_ma + 1e-10)
    chop_filter = bb_width_percentile > 0.5  # Avoid low volatility squeeze
    chop_filter_aligned = align_htf_to_ltf(prices, df_1d, chop_filter)
    
    # 12h Williams Alligator (Jaw=13*8, Teeth=8*5, Lips=5*3) smoothed with SMMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:  # Need enough for Alligator lips (5*3=13) + smoothing
        return np.zeros(n)
    
    median_12h = (df_12h['high'].values + df_12h['low'].values) / 2.0
    
    # Alligator Jaw: 13-period SMMA, shifted 8 bars
    jaw = pd.Series(median_12h).rolling(window=13, min_periods=13).mean().shift(8).values
    # Alligator Teeth: 8-period SMMA, shifted 5 bars
    teeth = pd.Series(median_12h).rolling(window=8, min_periods=8).mean().shift(5).values
    # Alligator Lips: 5-period SMMA, shifted 3 bars
    lips = pd.Series(median_12h).rolling(window=5, min_periods=5).mean().shift(3).values
    
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Alligator Sleep condition: all lines intertwined (max-min < 0.1% of price)
    alligator_max = np.maximum(np.maximum(jaw_aligned, teeth_aligned), lips_aligned)
    alligator_min = np.minimum(np.minimum(jaw_aligned, teeth_aligned), lips_aligned)
    alligator_range = alligator_max - alligator_min
    alligator_sleep = alligator_range < (0.001 * close)  # 0.1% threshold
    
    # Breakout conditions: price breaks above/below Alligator extremes
    breakout_up = high > alligator_max
    breakout_down = low < alligator_min
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(50, 34, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(chop_filter_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1d EMA50
        bullish_bias = curr_close > ema_1d_aligned[i]
        bearish_bias = curr_close < ema_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: Alligator breakout after sleep + trend + volume + chop filter
            # Long: price breaks above Alligator max AND was sleeping AND bullish bias AND volume spike AND chop filter
            long_entry = breakout_up[i] and alligator_sleep[i] and bullish_bias and vol_spike and chop_filter_aligned[i]
            # Short: price breaks below Alligator min AND was sleeping AND bearish bias AND volume spike AND chop filter
            short_entry = breakout_down[i] and alligator_sleep[i] and bearish_bias and vol_spike and chop_filter_aligned[i]
            
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
            # Exit: price falls below Alligator teeth OR loss of bullish bias
            if (curr_low < teeth_aligned[i]) or (curr_close < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above Alligator teeth OR loss of bearish bias
            if (curr_high > teeth_aligned[i]) or (curr_close > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_Breakout_1dEMA50_Trend_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0