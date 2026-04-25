#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeSpike_ATRStop_v3
Hypothesis: Donchian(20) breakout on 4h with 1d EMA34 trend filter and volume spike provides robust signals in both bull and bear markets. Uses ATR-based stoploss (signal=0 when price < highest - 2.5*ATR for longs, price > lowest + 2.5*ATR for shorts) to limit drawdown. Discrete position sizing (0.30) reduces fee churn. Target: 20-50 trades/year per symbol.
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
    
    # 1d data for trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 4h Donchian channels (20-period)
    # Using rolling window on high/low
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h ATR for volatility and stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h volume spike: current volume > 2.0 * 20-period volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # track highest high since entry for trailing stop
    lowest_since_entry = 0.0   # track lowest low since entry for trailing stop
    
    # Start index: need Donchian (20), ATR (14), volume MA (20), aligned EMA
    start_idx = max(20, 14, 20, 0)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper with volume spike and 1d uptrend
            long_breakout = (curr_close > highest_20[i]) and vol_spike[i] and (curr_close > ema_34_1d_aligned[i])
            # Short: price breaks below Donchian lower with volume spike and 1d downtrend
            short_breakout = (curr_close < lowest_20[i]) and vol_spike[i] and (curr_close < ema_34_1d_aligned[i])
            
            if long_breakout:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_high
            elif short_breakout:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
                lowest_since_entry = curr_low
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.30
            # Update highest high since entry
            if curr_high > highest_since_entry:
                highest_since_entry = curr_high
            # ATR-based stoploss: exit if price drops below highest - 2.5*ATR
            if curr_close < (highest_since_entry - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            # Also exit if price breaks below Donchian lower (failed breakout)
            elif curr_close < lowest_20[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            # Update lowest low since entry
            if curr_low < lowest_since_entry:
                lowest_since_entry = curr_low
            # ATR-based stoploss: exit if price rises above lowest + 2.5*ATR
            if curr_close > (lowest_since_entry + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            # Also exit if price breaks above Donchian upper (failed breakout)
            elif curr_close > highest_20[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeSpike_ATRStop_v3"
timeframe = "4h"
leverage = 1.0