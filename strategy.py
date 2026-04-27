#!/usr/bin/env python3
"""
4h_KAMA_Turn_With_Direction_Filter_v1
Hypothesis: KAMA adapts to market noise - turning points signal trend changes. 
Only take trades when aligned with 1d EMA34 trend and volume confirmation to avoid whipsaws.
KAMA turning point defined as: price crosses above/below KAMA with confirmation candle.
Designed for fewer trades (target 50-150/year) to reduce fee drag while capturing strong moves.
Works in both bull (follows trend) and bear (avoids counter-trend) via 1d trend filter.
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
    
    # Calculate KAMA (adaptive moving average)
    # ER = Efficiency Ratio = |net change| / sum(|changes|)
    # Smoothing constant = [ER * (fastest - slowest) + slowest]^2
    change = abs(np.diff(close, prepend=close[0]))
    abs_change = change
    er_num = abs(np.subtract(close, np.roll(close, 1)))
    er_den = np.sum(np.lib.stride_tricks.sliding_window_view(change, 10), axis=1)
    # Handle edge cases for ER calculation
    er = np.zeros_like(close)
    er[10:] = er_num[10:] / er_den
    er = np.where(er_den == 0, 0, er)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # for EMA 2
    slow_sc = 2 / (30 + 1)  # for EMA 30
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_avg)
    
    # Align all indicators to primary timeframe (4h)
    kama_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kama)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital (reduces drawdown)
    
    # Warmup: need KAMA calc (10), EMA34 (34), volume avg (20)
    start_idx = max(10, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kama_val = kama_aligned[i]
        ema34 = ema34_1d_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        
        # KAMA turn signals: price crosses KAMA with confirmation
        kama_cross_up = close_val > kama_val and close >= kama_val and close[i-1] <= kama_aligned[i-1]
        kama_cross_down = close_val < kama_val and close <= kama_val and close[i-1] >= kama_aligned[i-1]
        
        if position == 0:
            # Determine trend alignment: price vs EMA34 (1d)
            uptrend = close_val > ema34
            downtrend = close_val < ema34
            
            if uptrend and vol_conf and kama_cross_up:
                # Long: price crosses above KAMA in uptrend with volume
                signals[i] = size
                position = 1
                entry_price = close_val
            elif downtrend and vol_conf and kama_cross_down:
                # Short: price crosses below KAMA in downtrend with volume
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit: price crosses below KAMA or trend change
            if close_val < kama_val or close_val < ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: price crosses above KAMA or trend change
            if close_val > kama_val or close_val > ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_KAMA_Turn_With_Direction_Filter_v1"
timeframe = "4h"
leverage = 1.0