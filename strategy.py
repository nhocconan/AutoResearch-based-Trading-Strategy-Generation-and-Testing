#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_ATRStop_v1
Hypothesis: On 4h timeframe, trade breakouts above/below 20-period Donchian channels only when aligned with 1d EMA50 trend and confirmed by volume spike (>2.0x 20-bar average). Uses ATR(14) stoploss at 2.0x ATR. Discrete sizing at 0.25 to limit fee drag. Target: 15-40 trades/year on BTC/ETH/SOL. Donchian breakouts capture momentum, 1d EMA50 filters counter-trend moves, volume spike confirms institutional interest, ATR stop manages risk in volatile markets. Works in bull (breakouts continue) and bear (stoploss prevents large losses during reversals).
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR for stoploss (4h ATR)
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # first bar
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of EMA50 (50), ATR (14), volume MA (20), Donchian (20)
    start_idx = max(50, 14, 20, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_val = ema_50_1d_aligned[i]
        close_val = close[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        highest_high_val = highest_high[i]
        lowest_low_val = lowest_low[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian, above 1d EMA50, with volume spike
            long_signal = (close_val > highest_high_val) and (close_val > ema_50_val) and vol_spike
            
            # Short: price breaks below lower Donchian, below 1d EMA50, with volume spike
            short_signal = (close_val < lowest_low_val) and (close_val < ema_50_val) and vol_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below lower Donchian OR ATR stoploss (2.0*ATR below entry)
            if (close_val < lowest_low_val) or (close_val < entry_price - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above upper Donchian OR ATR stoploss (2.0*ATR above entry)
            if (close_val > highest_high_val) or (close_val > entry_price + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_ATRStop_v1"
timeframe = "4h"
leverage = 1.0