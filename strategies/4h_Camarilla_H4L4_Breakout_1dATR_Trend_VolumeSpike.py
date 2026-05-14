#!/usr/bin/env python3
"""
4h_Camarilla_H4L4_Breakout_1dATR_Trend_VolumeSpike
Hypothesis: On 4h timeframe, Camarilla H4/L4 breakouts with 1d ATR-based trend filter and volume spike.
Uses H4/L4 levels (wider than H3/L3) for fewer, higher-quality breakouts. Volume spike confirms institutional participation.
1d ATR trend filter ensures trades align with strong daily momentum (price > EMA + 0.5*ATR for long, < EMA - 0.5*ATR for short).
This filters out weak breakouts in choppy markets while capturing strong trending moves. Target: 20-50 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average with min_periods"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate Average True Range with min_periods"""
    if len(high) < period:
        return np.full_like(high, np.nan)
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff()
    tr3 = pd.Series(close).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for EMA34 and ATR14 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 trend filter
    ema_34_1d = calculate_ema(df_1d['close'].values, 34)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d ATR14 for trend strength filter
    atr_14_1d = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 1d data for Camarilla pivots (H4/L4 levels - wider bands for fewer trades)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    camarilla_range = 1.1 * (prev_high - prev_low)
    h4 = prev_close + camarilla_range * 0.50  # H4 level (widest)
    l4 = prev_close - camarilla_range * 0.50  # L4 level (widest)
    
    # Align Camarilla levels to 4h timeframe (completed 1d bar)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA (34) + ATR (14) + volume MA (20)
    start_idx = max(34, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla H4/L4 breakout + volume spike + 1d ATR trend alignment
            long_breakout = curr_high > h4_aligned[i]
            short_breakout = curr_low < l4_aligned[i]
            
            # Trend filter: price must be beyond EMA by 0.5*ATR to confirm strong momentum
            long_trend = curr_close > (ema_34_1d_aligned[i] + 0.5 * atr_14_1d_aligned[i])
            short_trend = curr_close < (ema_34_1d_aligned[i] - 0.5 * atr_14_1d_aligned[i])
            
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
            # Long position: exit when price closes below H4 (failed breakout) or trend weakens
            if curr_close < h4_aligned[i] or curr_close < (ema_34_1d_aligned[i] + 0.5 * atr_14_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes above L4 (failed breakout) or trend weakens
            if curr_close > l4_aligned[i] or curr_close > (ema_34_1d_aligned[i] - 0.5 * atr_14_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H4L4_Breakout_1dATR_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0