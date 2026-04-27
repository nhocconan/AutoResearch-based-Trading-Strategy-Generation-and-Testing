#!/usr/bin/env python3
"""
4h_KAMA_Trend_Donchian20_VolumeBreakout
Hypothesis: On 4h timeframe, use KAMA(10) for adaptive trend direction, Donchian(20) breakout for entry timing, and volume confirmation (>1.5x 20-period average) to filter false breakouts. Long when price breaks above Donchian upper band AND KAMA slope positive AND volume confirmed. Short when price breaks below Donchian lower band AND KAMA slope negative AND volume confirmed. Exit on opposite Donchian band touch or trend reversal. Designed for 4h to achieve 20-50 trades/year with low fee drag. KAMA adapts to market noise, reducing whipsaws in sideways markets while capturing trends in bull/bear regimes.
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
    
    # KAMA(10) for trend direction
    close_s = pd.Series(close)
    # Efficiency Ratio: ER = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(close - np.roll(close, 10))
    volatility = np.sum(np.abs(np.diff(close, axis=0))) if len(close) > 1 else 0
    # Vectorized ER calculation
    er = np.zeros_like(close)
    for i in range(10, len(close)):
        if i >= 10:
            price_change = np.abs(close[i] - close[i-10])
            sum_vol = np.sum(np.abs(np.diff(close[i-10:i+1])))
            er[i] = price_change / (sum_vol + 1e-10)
    # Smoothing constants: fastest SC = 2/(2+1)=0.667, slowest SC = 2/(30+1)=0.0645
    sc = (er * 0.603 + 0.0645) ** 2  # SC = [ER*(fastest-sc) + slowest-sc]^2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama_slope = kama - np.roll(kama, 1)
    kama_slope[0] = 0
    
    # Donchian(20) channels
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need KAMA (10), Donchian (20), volume avg (20)
    start_idx = max(20, 10)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_slope[i]) or np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        slope_val = kama_slope[i]
        upper_band = high_max[i]
        lower_band = low_min[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Donchian breakout with KAMA trend filter AND volume
            # Long: price breaks above upper band AND KAMA slope positive AND volume confirmed
            long_condition = (close_val > upper_band) and (slope_val > 0) and vol_conf
            # Short: price breaks below lower band AND KAMA slope negative AND volume confirmed
            short_condition = (close_val < lower_band) and (slope_val < 0) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price touches lower band OR trend reverses
            exit_condition = (close_val < lower_band) or (slope_val <= 0)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price touches upper band OR trend reverses
            exit_condition = (close_val > upper_band) or (slope_val >= 0)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_KAMA_Trend_Donchian20_VolumeBreakout"
timeframe = "4h"
leverage = 1.0