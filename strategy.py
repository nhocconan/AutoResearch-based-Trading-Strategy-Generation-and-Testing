#!/usr/bin/env python3
"""
6h_Aroon_Trend_1wADX_Filter
Hypothesis: Aroon identifies trend strength and direction (new highs/lows over 25 periods); combined with weekly ADX filter (>25) to ensure trending market, avoiding whipsaws in ranging markets. Works in bull markets via Aroon-up strength and in bear markets via Aroon-down strength. Targets ~15-25 trades/year on 6h to minimize fee drag.
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
    
    # Get weekly data for ADX filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly ADX (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w - np.roll(low_1w, 1))
    tr2 = np.abs(low_1w - np.roll(high_1w, 1))
    tr3 = np.abs(high_1w - np.roll(high_1w, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = high_1w[0] - low_1w[0]
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth over 14 periods
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = np.where(tr14 != 0, 100 * dm_plus14 / tr14, 0)
    di_minus = np.where(tr14 != 0, 100 * dm_minus14 / tr14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1w = adx  # already weekly
    
    # Align weekly ADX to 6h
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Aroon (25-period) on 6h data
    # Aroon Up: ((25 - periods since 25-period high) / 25) * 100
    # Aroon Down: ((25 - periods since 25-period low) / 25) * 100
    aroon_period = 25
    aroon_up = np.full(n, np.nan)
    aroon_down = np.full(n, np.nan)
    
    for i in range(aroon_period - 1, n):
        window_high = high[i - aroon_period + 1:i + 1]
        window_low = low[i - aroon_period + 1:i + 1]
        periods_since_high = aroon_period - 1 - np.argmax(window_high)
        periods_since_low = aroon_period - 1 - np.argmin(window_low)
        aroon_up[i] = ((aroon_period - periods_since_high) / aroon_period) * 100
        aroon_down[i] = ((aroon_period - periods_since_low) / aroon_period) * 100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for Aroon and weekly ADX
    start_idx = max(30, aroon_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(aroon_up[i]) or np.isnan(aroon_down[i]) or 
            np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        aroon_up_val = aroon_up[i]
        aroon_down_val = aroon_down[i]
        adx_val = adx_1w_aligned[i]
        
        # Only trade when weekly ADX > 25 (trending market)
        if adx_val > 25:
            if position == 0:
                # Long: Aroon Up > Aroon Down (bullish strength)
                if aroon_up_val > aroon_down_val:
                    signals[i] = size
                    position = 1
                # Short: Aroon Down > Aroon Up (bearish strength)
                elif aroon_down_val > aroon_up_val:
                    signals[i] = -size
                    position = -1
                else:
                    signals[i] = 0.0
            elif position == 1:
                # Exit long: Aroon Down > Aroon Up (trend weakness)
                if aroon_down_val > aroon_up_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = size
            elif position == -1:
                # Exit short: Aroon Up > Aroon Down (trend weakness)
                if aroon_up_val > aroon_down_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -size
        else:
            # Range market: stay flat
            signals[i] = 0.0
            position = 0
    
    return signals

name = "6h_Aroon_Trend_1wADX_Filter"
timeframe = "6h"
leverage = 1.0