#!/usr/bin/env python3
"""
6h_12h_PriceAction_Volume_Structure
Hypothesis: Combine price action structure (higher highs/lows) with volume confirmation on 6h, filtered by 12h trend.
- Long when: Higher low forms (bullish structure), volume > 20-period average, price above 12h EMA50
- Short when: Lower high forms (bearish structure), volume > 20-period average, price below 12h EMA50
- Exit when: Structure breaks or trend reverses
This captures institutional accumulation/distribution patterns. Volume confirms participation.
Structure filters out chop. Targets 15-25 trades/year (60-100 over 4 years) to minimize fee drag.
Works in bull by buying dips in uptrend, in bear by selling rallies in downtrend.
"""

name = "6h_12h_PriceAction_Volume_Structure"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 6h OHLCV
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    volume_6h = prices['volume'].values
    
    # --- 12h Trend Filter: EMA50 ---
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # --- Structure Detection: Higher Lows / Lower Highs ---
    # Higher Low: current low > previous low AND previous low < low before that
    # Lower High: current high < previous high AND previous high > high before that
    hl_signal = np.zeros(n, dtype=bool)  # Higher Low detected
    lh_signal = np.zeros(n, dtype=bool)  # Lower High detected
    
    for i in range(2, n):
        # Higher Low: low[i] > low[i-1] and low[i-1] < low[i-2]
        if low_6h[i] > low_6h[i-1] and low_6h[i-1] < low_6h[i-2]:
            hl_signal[i] = True
        # Lower High: high[i] < high[i-1] and high[i-1] > high[i-2]
        if high_6h[i] < high_6h[i-1] and high_6h[i-1] > high_6h[i-2]:
            lh_signal[i] = True
    
    # --- Volume Confirmation: 6h volume > 20-period average ---
    vol_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 20  # for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 12h trend
        trend_up = close_6h[i] > ema50_12h_aligned[i]
        trend_down = close_6h[i] < ema50_12h_aligned[i]
        
        # Volume confirmation
        vol_ok = volume_6h[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for entries only in direction of 12h trend with volume and structure
            if hl_signal[i] and trend_up and vol_ok:
                # Bullish structure + uptrend + volume = Long
                signals[i] = 0.25
                position = 1
            elif lh_signal[i] and trend_down and vol_ok:
                # Bearish structure + downtrend + volume = Short
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: bearish structure forms OR trend turns down
                if lh_signal[i] or not trend_up:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: bullish structure forms OR trend turns up
                if hl_signal[i] or not trend_down:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals