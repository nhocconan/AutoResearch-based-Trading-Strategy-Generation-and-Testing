#!/usr/bin/env python3
"""
1h_4h_1d_Squeeze_Breakout_Volume
Hypothesis: Combines Bollinger Band squeeze detection on 1d with breakout confirmation on 4h and precise entry on 1h.
In low volatility (BB width < 20th percentile), waits for 4h close outside Bollinger Bands with volume > 1.5x 20-period average.
Enters on 1h break of the 4h breakout candle's high/low with volume confirmation.
Works in both bull and bear markets by trading volatility expansion after contraction.
Target: 15-37 trades/year on 1h (60-150 total over 4 years).
"""

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
    
    # Get daily data for Bollinger Bands and squeeze detection
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Bollinger Bands (20, 2.0) on daily
    ma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std()
    upper_bb = ma_20 + (2.0 * std_20)
    lower_bb = ma_20 - (2.0 * std_20)
    bb_width = upper_bb - lower_bb
    
    # Calculate 20-period percentile of BB width for squeeze detection (20th percentile)
    bb_width_series = pd.Series(bb_width.values)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Squeeze condition: BB width < 20th percentile
    squeeze = bb_width_percentile < 20.0
    
    # Get 4h data for breakout direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h Bollinger Bands (20, 2.0)
    ma_20_4h = pd.Series(close_4h).rolling(window=20, min_periods=20).mean()
    std_20_4h = pd.Series(close_4h).rolling(window=20, min_periods=20).std()
    upper_bb_4h = ma_20_4h + (2.0 * std_20_4h)
    lower_bb_4h = ma_20_4h - (2.0 * std_20_4h)
    
    # 4h breakout conditions: close outside BB with volume expansion
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean()
    volume_expansion_4h = volume_4h > (vol_ma_20_4h * 1.5)
    breakout_up = (close_4h > upper_bb_4h) & volume_expansion_4h
    breakout_down = (close_4h < lower_bb_4h) & volume_expansion_4h
    
    # Align all signals to 1h timeframe
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze)
    breakout_up_aligned = align_htf_to_ltf(prices, df_4h, breakout_up)
    breakout_down_aligned = align_htf_to_ltf(prices, df_4h, breakout_down)
    upper_bb_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_bb_4h.values)
    lower_bb_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_bb_4h.values)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.20  # 20% of capital
    
    # Track breakout candle high/low for entry
    breakout_high_level = np.zeros(n)
    breakout_low_level = np.zeros(n)
    
    for i in range(50, n):
        # Skip if not in session or data not ready
        if not session_mask[i] or \
           np.isnan(squeeze_aligned[i]) or \
           np.isnan(breakout_up_aligned[i]) or \
           np.isnan(breakout_down_aligned[i]) or \
           np.isnan(upper_bb_4h_aligned[i]) or \
           np.isnan(lower_bb_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Update breakout levels when new 4h breakout occurs
        if breakout_up_aligned[i]:
            breakout_high_level[i] = upper_bb_4h_aligned[i]
            breakout_low_level[i] = breakout_low_level[i-1] if i > 0 else 0
        elif breakout_down_aligned[i]:
            breakout_low_level[i] = lower_bb_4h_aligned[i]
            breakout_high_level[i] = breakout_high_level[i-1] if i > 0 else 0
        else:
            # Carry forward levels
            breakout_high_level[i] = breakout_high_level[i-1] if i > 0 else 0
            breakout_low_level[i] = breakout_low_level[i-1] if i > 0 else 0
        
        # Entry conditions: squeeze active and price breaks 4h breakout level with volume
        if squeeze_aligned[i]:
            # Volume confirmation on 1h
            vol_ma_20_1h = pd.Series(volume[:i+1]).rolling(window=20, min_periods=20).mean().iloc[-1] if i >= 20 else 0
            volume_expansion_1h = volume[i] > (vol_ma_20_1h * 1.5) if i >= 20 else False
            
            # Long entry: price breaks above 4h breakout high
            if breakout_high_level[i] > 0 and close[i] > breakout_high_level[i] and volume_expansion_1h:
                if position != 1:
                    position = 1
                    signals[i] = position_size
                else:
                    signals[i] = position_size
            # Short entry: price breaks below 4h breakout low
            elif breakout_low_level[i] > 0 and close[i] < breakout_low_level[i] and volume_expansion_1h:
                if position != -1:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = -position_size
            # Hold or flat
            elif position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        else:
            # No squeeze - exit any position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_Squeeze_Breakout_Volume"
timeframe = "1h"
leverage = 1.0