#!/usr/bin/env python3
"""
Experiment #2894: 1h Donchian Breakout + 4h/1d Trend Filter + Volume Spike
HYPOTHESIS: Donchian(20) breakouts on 1h timeframe capture short-term trends with precise entry.
4h EMA(50) and 1d EMA(200) provide multi-timeframe directional bias: only take longs when both
HTFs are bullish (price > EMA), shorts when both bearish. Volume spike (>1.5x 20-period average)
confirms breakout strength. Session filter (08-20 UTC) reduces noise. Target: 60-150 total trades
over 4 years = 15-37/year for 1h. Position size fixed at 0.20 to minimize fee impact.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2894_1h_donchian20_4h_1d_ema_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours for filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h EMA(50) for medium-term trend (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === HTF: 1d EMA(200) for long-term trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 1h Indicators: Donchian channels (20-period) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Fixed 20% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    
    warmup = max(200, 50, 20, 20)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # Skip if any data invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Exit logic: reverse signal or Donchian mean reversion
        if in_position:
            # Exit if price re-enters Donchian channel (take profit/mean reversion)
            if position_side > 0 and price <= highest_high[i]:
                in_position = False
                position_side = 0
                signals[i] = 0.0
            elif position_side < 0 and price >= lowest_low[i]:
                in_position = False
                position_side = 0
                signals[i] = 0.0
            else:
                # Hold position
                signals[i] = SIZE if position_side > 0 else -SIZE
            continue
        
        # New position entry: require volume spike and HTF alignment
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Check HTF trend alignment: both 4h and 1d EMAs must agree
            bullish_alignment = price > ema_4h_aligned[i] and price > ema_1d_aligned[i]
            bearish_alignment = price < ema_4h_aligned[i] and price < ema_1d_aligned[i]
            
            # Long entry: Donchian breakout + bullish HTF alignment
            if price > highest_high[i] and bullish_alignment:
                in_position = True
                position_side = 1
                signals[i] = SIZE
            # Short entry: Donchian breakdown + bearish HTF alignment
            elif price < lowest_low[i] and bearish_alignment:
                in_position = True
                position_side = -1
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals