#!/usr/bin/env python3
# 12h_TripleConfirmation_RangeBreakout
# Hypothesis: In ranging markets (low volatility), price tends to revert to mean.
# When volatility expands (high volatility), breakouts from Bollinger Bands with
# volume confirmation and ADX trend strength provide high-probability entries.
# Works in both bull and bear markets by adapting to volatility regime.

name = "12h_TripleConfirmation_RangeBreakout"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Bollinger Bands and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Bollinger Bands (20, 2.0)
    bb_length = 20
    bb_mult = 2.0
    bb_src = df_1d['close'].values
    bb_basis = pd.Series(bb_src).rolling(window=bb_length, min_periods=bb_length).mean().values
    bb_dev = bb_mult * pd.Series(bb_src).rolling(window=bb_length, min_periods=bb_length).std().values
    bb_upper = bb_basis + bb_dev
    bb_lower = bb_basis - bb_dev
    
    # ADX (14) for trend strength
    adx_len = 14
    tr1 = pd.Series(df_1d['high']).rolling(2).max().values - pd.Series(df_1d['low']).rolling(2).min().values
    tr2 = abs(pd.Series(df_1d['high']).rolling(2).max().values - pd.Series(df_1d['close']).shift(1).values)
    tr3 = abs(pd.Series(df_1d['low']).rolling(2).min().values - pd.Series(df_1d['close']).shift(1).values)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    plus_dm = pd.Series(df_1d['high']).diff().values
    minus_dm = pd.Series(df_1d['low']).diff().values * -1
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    tr_ma = pd.Series(tr).rolling(window=adx_len, min_periods=adx_len).mean().values
    plus_di_ma = 100 * (pd.Series(plus_dm).rolling(window=adx_len, min_periods=adx_len).mean().values / tr_ma)
    minus_di_ma = 100 * (pd.Series(minus_dm).rolling(window=adx_len, min_periods=adx_len).mean().values / tr_ma)
    dx = 100 * abs(plus_di_ma - minus_di_ma) / (plus_di_ma + minus_di_ma + 1e-10)
    adx = pd.Series(dx).rolling(window=adx_len, min_periods=adx_len).mean().values
    
    # Align indicators to 12h timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation (20-period MA on 12h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Bollinger Bands (20), ADX (14+14), volume MA (20)
    start_idx = max(bb_length, adx_len*2, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(bb_upper_aligned[i]) or 
            np.isnan(bb_lower_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility expansion: Bollinger Band width increasing
        bb_width = bb_upper_aligned[i] - bb_lower_aligned[i]
        bb_width_prev = bb_upper_aligned[i-1] - bb_lower_aligned[i-1] if i > 0 else bb_width
        volatility_expanding = bb_width > bb_width_prev * 1.05
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Trend strength filter
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Long entry: volatility expanding + price breaks above upper BB + volume + strong trend
            if volatility_expanding and close[i] > bb_upper_aligned[i] and volume_confirm and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short entry: volatility expanding + price breaks below lower BB + volume + strong trend
            elif volatility_expanding and close[i] < bb_lower_aligned[i] and volume_confirm and strong_trend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: volatility contracting or price re-enters below upper BB
            bb_width_current = bb_upper_aligned[i] - bb_lower_aligned[i]
            bb_width_prev_exit = bb_upper_aligned[i-1] - bb_lower_aligned[i-1] if i > 0 else bb_width_current
            volatility_contracting = bb_width_current < bb_width_prev_exit * 0.95
            
            if volatility_contracting or close[i] < bb_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: volatility contracting or price re-enters above lower BB
            bb_width_current = bb_upper_aligned[i] - bb_lower_aligned[i]
            bb_width_prev_exit = bb_upper_aligned[i-1] - bb_lower_aligned[i-1] if i > 0 else bb_width_current
            volatility_contracting = bb_width_current < bb_width_prev_exit * 0.95
            
            if volatility_contracting or close[i] > bb_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals