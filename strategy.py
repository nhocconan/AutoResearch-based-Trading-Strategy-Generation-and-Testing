#!/usr/bin/env python3
"""
12h Donchian20 Breakout + 1d EMA50 Trend + Volume Spike + Chop Filter
Hypothesis: Donchian channel breakouts on 12h capture medium-term momentum. 
1d EMA50 provides trend filter to avoid counter-trend trades. Volume confirmation 
ensures breakout validity. Chop filter (BBW percentile > 0.5) avoids low-volatility 
false breakouts. Designed for 50-150 trades over 4 years on 12h timeframe.
Works in bull/bear via trend filter and regime avoidance.
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
    
    # Load 12h data ONCE before loop for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Donchian channels for each 12h bar (20-period)
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to avoid look-ahead (use previous bar's levels)
    donchian_high_shifted = np.roll(donchian_high, 1)
    donchian_low_shifted = np.roll(donchian_low, 1)
    donchian_high_shifted[0] = np.nan
    donchian_low_shifted[0] = np.nan
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high_shifted)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low_shifted)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(50, 20, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(chop_filter_aligned[i]) or 
            np.isnan(vol_ma[i])):
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
            # Look for entry signals - require: Donchian breakout + trend + volume + chop filter
            # Long: price breaks above Donchian high AND bullish bias AND volume spike AND chop filter
            long_entry = (curr_high > donchian_high_aligned[i]) and bullish_bias and vol_spike and chop_filter_aligned[i]
            # Short: price breaks below Donchian low AND bearish bias AND volume spike AND chop filter
            short_entry = (curr_low < donchian_low_aligned[i]) and bearish_bias and vol_spike and chop_filter_aligned[i]
            
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
            # Exit: price falls below Donchian low (reversal) OR loss of bullish bias
            if (curr_low < donchian_low_aligned[i]) or (curr_close < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above Donchian high (reversal) OR loss of bearish bias
            if (curr_high > donchian_high_aligned[i]) or (curr_close > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0