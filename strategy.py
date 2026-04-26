#!/usr/bin/env python3
"""
6h_KeltnerDonchian_HybridBreakout_1dTrend_v1
Hypothesis: Combines Keltner Channel (volatility-based) and Donchian Channel (price-based) breakouts on 6h with 1d EMA50 trend filter. Long when price breaks above upper Keltner AND Donchian with volume confirmation in uptrend. Short when breaks below lower Keltner AND Donchian with volume confirmation in downtrend. Uses discrete positions (0.0, ±0.25) to minimize fee churn. Works in bull/bear by following 1d trend while capturing momentum bursts validated by volume.
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
    
    # Load 1d data ONCE before loop for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Keltner Channel (20, 2.0) on 6h
    atr_period = 20
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    kc_upper = ema20 + (2.0 * atr)
    kc_lower = ema20 - (2.0 * atr)
    
    # Donchian Channel (20) on 6h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    volume_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirm = volume > (volume_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1d trend filter
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        # Long conditions: price breaks above BOTH KC upper AND Donchian upper with volume confirmation in uptrend
        long_breakout = (close[i] > kc_upper[i]) and (close[i] > highest_high[i]) and volume_confirm[i] and uptrend
        # Short conditions: price breaks below BOTH KC lower AND Donchian lower with volume confirmation in downtrend
        short_breakout = (close[i] < kc_lower[i]) and (close[i] < lowest_low[i]) and volume_confirm[i] and downtrend
        
        # Exit conditions: price returns to middle of KC or trend weakens
        long_exit = (close[i] < ema20[i]) or (not uptrend)
        short_exit = (close[i] > ema20[i]) or (not downtrend)
        
        if long_breakout and position != 1:
            signals[i] = 0.25
            position = 1
        elif short_breakout and position != -1:
            signals[i] = -0.25
            position = -1
        elif position == 1 and long_exit:
            signals[i] = 0.0
            position = 0
        elif position == -1 and short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_KeltnerDonchian_HybridBreakout_1dTrend_v1"
timeframe = "6h"
leverage = 1.0