#!/usr/bin/env python3
"""
12h Bollinger Band Squeeze Breakout with Volume and Trend Filter
Long: Price breaks above upper BB(20,2) + volume > 1.5x 12h volume MA + price > 1D EMA50
Short: Price breaks below lower BB(20,2) + volume > 1.5x 12h volume MA + price < 1D EMA50
Exit: Opposite touch of middle BB or volatility expansion (BB width > 1.5x 20-period avg width)
Targets 20-30 trades/year per symbol by requiring BB squeeze precondition.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1D data for trend filter and BB calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1D Bollinger Bands (20,2)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    middle_bb = sma_20
    bb_width = upper_bb - lower_bb
    
    # Squeeze condition: BB width < 1.5x 20-period average width
    bb_width_ma20 = pd.Series(bb_width).rolling(window=20, min_periods=20).mean()
    squeeze_condition = bb_width < 1.5 * bb_width_ma20
    
    # 1D EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean()
    
    # Align 1D indicators to 12h
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb.values)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb.values)
    middle_bb_aligned = align_htf_to_ltf(prices, df_1d, middle_bb.values)
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze_condition.values)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d.values)
    
    # 12h volume moving average (24-period for confirmation)
    df_12h = get_htf_data(prices, '12h')
    volume_ma_24 = pd.Series(df_12h['volume']).rolling(window=24, min_periods=24).mean()
    volume_ma_24_12h = align_htf_to_ltf(prices, df_12h, volume_ma_24.values)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 60  # warmup for BB calculations
    
    for i in range(start_idx, n):
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(middle_bb_aligned[i]) or np.isnan(squeeze_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_24_12h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_24_12h[i]
        
        if position == 0:
            # Long: BB squeeze breakout above upper BB + volume + trend
            if (price > upper_bb_aligned[i] and squeeze_aligned[i] and 
                vol > 1.5 * vol_ma and price > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: BB squeeze breakout below lower BB + volume + trend
            elif (price < lower_bb_aligned[i] and squeeze_aligned[i] and 
                  vol > 1.5 * vol_ma and price < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price touches middle BB OR volatility expansion (BB width > 1.5x avg)
            bb_width_current = upper_bb_aligned[i] - lower_bb_aligned[i]
            bb_width_ma20_current = bb_width_ma20.iloc[i] if hasattr(bb_width_ma20, 'iloc') else bb_width_ma20[i]
            if (price <= middle_bb_aligned[i] or 
                bb_width_current > 1.5 * bb_width_ma20_current):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches middle BB OR volatility expansion
            bb_width_current = upper_bb_aligned[i] - lower_bb_aligned[i]
            bb_width_ma20_current = bb_width_ma20.iloc[i] if hasattr(bb_width_ma20, 'iloc') else bb_width_ma20[i]
            if (price >= middle_bb_aligned[i] or 
                bb_width_current > 1.5 * bb_width_ma20_current):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_BB_Squeeze_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0