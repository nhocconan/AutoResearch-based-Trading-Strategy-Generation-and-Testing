#!/usr/bin/env python3
"""
12h Camarilla H3/L3 Breakout + 1d EMA34 Trend + Volume Spike + Chop Filter
Hypothesis: Camarilla H3/L3 levels act as strong intraday support/resistance on 12h timeframe.
Breakouts above H3 (long) or below L3 (short) capture momentum with trend filter (1d EMA34).
Volume confirmation ensures breakout validity. Chop filter (BBW percentile > 0.5) avoids low-volatility false breakouts.
Discrete sizing (0.25) limits drawdown. Designed for 50-150 trades over 4 years on 12h.
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
    
    # Load 1d data ONCE before loop for EMA34 trend filter and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
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
    
    # Load 12h data ONCE before loop for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for each 12h bar: H3, L3
    # H3 = close + (high - low) * 1.1/4
    # L3 = close - (high - low) * 1.1/4
    rng = high_12h - low_12h
    camarilla_h3 = close_12h + rng * (1.1 / 4)
    camarilla_l3 = close_12h - rng * (1.1 / 4)
    
    # Shift by 1 to avoid look-ahead (use previous bar's levels)
    camarilla_h3_shifted = np.roll(camarilla_h3, 1)
    camarilla_l3_shifted = np.roll(camarilla_l3, 1)
    camarilla_h3_shifted[0] = np.nan
    camarilla_l3_shifted[0] = np.nan
    
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3_shifted)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3_shifted)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(34, 20, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(chop_filter_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1d EMA34
        bullish_bias = curr_close > ema_1d_aligned[i]
        bearish_bias = curr_close < ema_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla breakout + trend + volume + chop filter
            # Long: price breaks above Camarilla H3 AND bullish bias AND volume spike AND chop filter
            long_entry = (curr_high > camarilla_h3_aligned[i]) and bullish_bias and vol_spike and chop_filter_aligned[i]
            # Short: price breaks below Camarilla L3 AND bearish bias AND volume spike AND chop filter
            short_entry = (curr_low < camarilla_l3_aligned[i]) and bearish_bias and vol_spike and chop_filter_aligned[i]
            
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
            # Exit: price falls below Camarilla L3 (reversal) OR loss of bullish bias
            if (curr_low < camarilla_l3_aligned[i]) or (curr_close < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above Camarilla H3 (reversal) OR loss of bearish bias
            if (curr_high > camarilla_h3_aligned[i]) or (curr_close > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0