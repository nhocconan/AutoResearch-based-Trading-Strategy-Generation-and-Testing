#!/usr/bin/env python3
"""
4H_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels from 1d combined with 1d EMA34 trend and volume spike creates high-probability entries in both bull and bear markets. The Camarilla R1/S1 levels act as dynamic support/resistance, while the 1d EMA34 filter ensures alignment with longer-term momentum. Volume spike confirms conviction. Designed for 20-50 trades/year per symbol to minimize fee drag and achieve consistent performance across market regimes.
"""

name = "4H_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 1d Camarilla Pivot Levels (R1, S1) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar to close)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # --- 1d Trend Filter: EMA34 ---
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # --- Volume Filter: spike above 1.5x median of last 30 periods ---
    vol_median = pd.Series(volume_4h).rolling(window=30, min_periods=10).median().values
    vol_threshold = vol_median * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 30  # for volume median
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_threshold[i])):
            if position != 0:
                # Check stoploss
                if position == 1 and close_4h[i] <= entry_price - 2.0 * (high_4h[i] - low_4h[i]):
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_4h[i] >= entry_price + 2.0 * (high_4h[i] - low_4h[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine 1d trend
        trend_up = close_4h[i] > ema34_1d_aligned[i]
        trend_down = close_4h[i] < ema34_1d_aligned[i]
        
        # Volume filter: spike above 1.5x median
        vol_ok = volume_4h[i] > vol_threshold[i]
        
        if position == 0:
            # Look for entries only in direction of 1d trend with volume spike
            if close_4h[i] > r1_aligned[i] and trend_up and vol_ok:
                # Long: price breaks above R1 + 1d uptrend + volume spike
                signals[i] = 0.25
                position = 1
                entry_price = close_4h[i]
            elif close_4h[i] < s1_aligned[i] and trend_down and vol_ok:
                # Short: price breaks below S1 + 1d downtrend + volume spike
                signals[i] = -0.25
                position = -1
                entry_price = close_4h[i]
        else:
            # Update stoploss and check exits
            if position == 1:
                # Stoploss
                if close_4h[i] <= entry_price - 2.0 * (high_4h[i] - low_4h[i]):
                    signals[i] = 0.0
                    position = 0
                # Exit: price returns to or below pivot
                elif close_4h[i] <= pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Stoploss
                if close_4h[i] >= entry_price + 2.0 * (high_4h[i] - low_4h[i]):
                    signals[i] = 0.0
                    position = 0
                # Exit: price returns to or above pivot
                elif close_4h[i] >= pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals