#!/usr/bin/env python3
"""
12h Camarilla H3L3 Breakout with 1d Volume Spike and ADX Trend Filter
Hypothesis: Camarilla H3/L3 levels act as intraday support/resistance on 12h charts. 
Breakout above H3 with volume spike and bullish 1d ADX > 25 indicates strong upward momentum. 
Breakdown below L3 with volume spike and bearish 1d ADX > 25 indicates strong downward momentum.
Uses discrete sizing (0.0, ±0.30) to minimize fee churn. Target: 12-30 trades/year on 12h.
Works in bull markets (breakouts above H3) and bear markets (breakdowns below L3) by requiring 
1d ADX trend alignment, reducing false signals in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter and volume average (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).rolling(1).max() - pd.Series(low_1d).rolling(1).min()
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1d) - pd.Series(high_1d).shift(1)
    down_move = pd.Series(low_1d).shift(1) - pd.Series(low_1d)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean() / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean() / atr_1d
    dx_1d = 100 * abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 1d average volume for volume spike detection
    vol_ma_1d = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate Camarilla levels for 12h (using previous day's OHLC)
    # Camarilla levels are based on previous day's range
    # We need to align the previous day's close to current 12h bar
    prev_close_1d = pd.Series(close_1d).shift(1).values
    prev_high_1d = pd.Series(high_1d).shift(1).values
    prev_low_1d = pd.Series(low_1d).shift(1).values
    
    # Align previous day's OHLC to 12h timeframe
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close_1d)
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high_1d)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low_1d)
    
    # Calculate Camarilla levels
    range_prev = prev_high_aligned - prev_low_aligned
    h3 = prev_close_aligned + range_prev * 1.1 / 4
    l3 = prev_close_aligned - range_prev * 1.1 / 4
    h4 = prev_close_aligned + range_prev * 1.1 / 2
    l4 = prev_close_aligned - range_prev * 1.1 / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(h3[i]) or np.isnan(l3[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_h3 = h3[i]
        curr_l3 = l3[i]
        curr_h4 = h4[i]
        curr_l4 = l4[i]
        adx_trend = adx_1d_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        
        # Volume spike: current volume > 1.5 * 1d average volume
        volume_spike = curr_volume > 1.5 * vol_ma
        
        # Strong trend: ADX > 25
        strong_trend = adx_trend > 25
        
        if position == 0:
            # Look for entry signals
            # Long: Break above H3 with volume spike and strong uptrend
            long_entry = (curr_close > curr_h3) and volume_spike and strong_trend
            # Short: Break below L3 with volume spike and strong downtrend
            short_entry = (curr_close < curr_l3) and volume_spike and strong_trend
            
            if long_entry:
                signals[i] = 0.30
                position = 1
            elif short_entry:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Price falls below L3 (breakdown of opposite level) OR ADX < 20 (trend weakening)
            if (curr_close < curr_l3) or (adx_trend < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position management
            # Exit: Price rises above H3 (breakout of opposite level) OR ADX < 20 (trend weakening)
            if (curr_close > curr_h3) or (adx_trend < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dVolumeSpike_ADXTrend"
timeframe = "12h"
leverage = 1.0