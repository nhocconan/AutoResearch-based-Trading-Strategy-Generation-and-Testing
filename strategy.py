#!/usr/bin/env python3
"""
12h_1W_Camarilla_Pivot_Breakout_Trend_Filter_v1
Hypothesis: Trade weekly Camarilla pivot breakouts on 12h timeframe with weekly EMA50 trend filter and volume confirmation. Buy when price breaks above weekly H4 with volume > 2x 100-period average and weekly trend bullish, sell when price breaks below weekly L4 with volume confirmation and weekly trend bearish. Uses weekly trend to avoid counter-trend trades in choppy markets. Designed for low frequency (<30 trades/year) to minimize fee drag while capturing major trend continuations in both bull and bear markets.
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
    
    # Volume confirmation: current volume > 2x 100-period average
    vol_ma_100 = pd.Series(volume).rolling(window=100, min_periods=100).mean()
    volume_expansion = volume > (vol_ma_100 * 2.0)
    
    # Previous week's high/low/close for Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    prev_high_1w = df_1w['high'].values
    prev_low_1w = df_1w['low'].values
    prev_close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels
    camarilla_h4_1w = prev_close_1w + 1.1 * (prev_high_1w - prev_low_1w) / 2
    camarilla_l4_1w = prev_close_1w - 1.1 * (prev_high_1w - prev_low_1w) / 2
    
    # Align weekly levels to 12h timeframe (wait for weekly close)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4_1w)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4_1w)
    
    # Weekly EMA50 trend filter
    ema50_1w_raw = pd.Series(prev_close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w_raw)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    bars_since_entry = 0  # Track holding period
    
    for i in range(100, n):  # warmup period
        # Skip if any required data is not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        # Determine weekly trend: bullish if price > EMA50, bearish if price < EMA50
        weekly_trend_bullish = close[i] > ema50_1w_aligned[i]
        weekly_trend_bearish = close[i] < ema50_1w_aligned[i]
        
        # Long signal: break above weekly Camarilla H4 with volume expansion and weekly trend bullish
        long_signal = (close[i] > camarilla_h4_aligned[i] and 
                      volume_expansion[i] and 
                      weekly_trend_bullish)
        
        # Short signal: break below weekly Camarilla L4 with volume expansion and weekly trend bearish
        short_signal = (close[i] < camarilla_l4_aligned[i] and 
                       volume_expansion[i] and 
                       weekly_trend_bearish)
        
        # Exit conditions: minimum holding period reached and opposite signal
        if position == 1 and bars_since_entry >= 8 and short_signal:
            position = -1
            signals[i] = -position_size
            bars_since_entry = 0
        elif position == -1 and bars_since_entry >= 8 and long_signal:
            position = 1
            signals[i] = position_size
            bars_since_entry = 0
        elif position == 0:
            if long_signal:
                position = 1
                signals[i] = position_size
                bars_since_entry = 0
            elif short_signal:
                position = -1
                signals[i] = -position_size
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "12h_1W_Camarilla_Pivot_Breakout_Trend_Filter_v1"
timeframe = "12h"
leverage = 1.0