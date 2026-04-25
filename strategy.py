#!/usr/bin/env python3
"""
4h_Camarilla_H3L3_Breakout_1dATR_Trend_VolumeSpike
Hypothesis: On 4h timeframe, Camarilla H3/L3 breakout from prior 1d bar, combined with 1d ATR trend filter (price > SMA50 + 0.5*ATR for long, < SMA50 - 0.5*ATR for short) and volume confirmation.
The ATR-adjusted trend filter adapts to volatility, reducing whipsaw in choppy markets while maintaining trend alignment.
Volume spike confirms institutional participation.
Designed for 19-50 trades/year (75-200 over 4 years) to minimize fee drag.
Works in bull markets via breakout continuation and bear markets via volatility-adjusted trend following.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla calculation and ATR (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Prior 1d bar OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: H3, L3 (strong intraday support/resistance)
    camarilla_range = prev_high - prev_low
    h3 = prev_close + camarilla_range * 1.1 / 4
    l3 = prev_close - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (completed 1d bar)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # 1d ATR(14) for volatility-adjusted trend filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    
    # 1d SMA50 for trend base
    sma_50 = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    
    # ATR-adjusted trend levels: long when price > SMA50 + 0.5*ATR, short when price < SMA50 - 0.5*ATR
    long_threshold = sma_50 + 0.5 * atr_14
    short_threshold = sma_50 - 0.5 * atr_14
    
    long_threshold_aligned = align_htf_to_ltf(prices, df_1d, long_threshold)
    short_threshold_aligned = align_htf_to_ltf(prices, df_1d, short_threshold)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Camarilla (1d shift), ATR (14), SMA (50), volume MA (20)
    start_idx = max(50, 20, 14)  # 50 covers all
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(long_threshold_aligned[i]) or np.isnan(short_threshold_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla H3/L3 breakout + volume spike + ATR-adjusted trend alignment
            long_breakout = curr_high > h3_aligned[i]
            short_breakout = curr_low < l3_aligned[i]
            
            # ATR-adjusted trend filter
            long_trend = curr_close > long_threshold_aligned[i]
            short_trend = curr_close < short_threshold_aligned[i]
            
            long_entry = (long_breakout and volume_spike[i] and long_trend)
            short_entry = (short_breakout and volume_spike[i] and short_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below Camarilla H3 (failed breakout) or below ATR-adjusted trend
            if curr_close < h3_aligned[i] or curr_close < long_threshold_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above Camarilla L3 (failed breakout) or above ATR-adjusted trend
            if curr_close > l3_aligned[i] or curr_close > short_threshold_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dATR_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0