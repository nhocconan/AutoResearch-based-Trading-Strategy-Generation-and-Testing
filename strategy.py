#!/usr/bin/env python3
"""
12h_Alligator_ElderRay_1wTrend
Hypothesis: Williams Alligator (13,8,5 SMAs) and Elder Ray (13-period bull/bear power) on 12h confirm trend when aligned with 1w EMA34. Enter long when price > Alligator jaw (13 SMA) and bull power > 0 with 1w uptrend; enter short when price < Alligator lips (5 SMA) and bear power < 0 with 1w downtrend. Uses volume confirmation (1.5x median) to avoid false signals. Designed for low-frequency, high-conviction trades in both bull and bear markets by requiring multi-timeframe alignment.
"""

name = "12h_Alligator_ElderRay_1wTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    volume_12h = prices['volume'].values
    
    # --- 1w Trend Filter: EMA34 ---
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # --- Williams Alligator (13,8,5 SMAs) ---
    # Jaw (13-period SMA)
    jaw = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().values
    # Teeth (8-period SMA)
    teeth = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().values
    # Lips (5-period SMA)
    lips = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().values
    
    # --- Elder Ray (13-period Bull/Bear Power) ---
    ema13 = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_12h - ema13  # High minus EMA13
    bear_power = low_12h - ema13   # Low minus EMA13
    
    # --- Volume Filter: spike above 1.5x median of last 20 periods ---
    vol_median = pd.Series(volume_12h).rolling(window=20, min_periods=10).median().values
    vol_threshold = vol_median * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period (max of 13, 8, 5, 20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_threshold[i])):
            if position != 0:
                # Check stoploss (2x ATR approximation using high-low range)
                hl_range = pd.Series(high_12h - low_12h).rolling(window=10, min_periods=10).mean().values
                if position == 1 and close_12h[i] <= entry_price - 2.0 * hl_range[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_12h[i] >= entry_price + 2.0 * hl_range[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine 1w trend
        trend_up = close_12h[i] > ema34_1w_aligned[i]
        trend_down = close_12h[i] < ema34_1w_aligned[i]
        
        # Volume filter: spike above 1.5x median
        vol_ok = volume_12h[i] > vol_threshold[i]
        
        if position == 0:
            # Look for entries only in direction of 1w trend with volume spike
            if close_12h[i] > jaw[i] and bull_power[i] > 0 and trend_up and vol_ok:
                # Long: price above Alligator jaw, bull power positive, 1w uptrend, volume spike
                signals[i] = 0.25
                position = 1
                entry_price = close_12h[i]
            elif close_12h[i] < lips[i] and bear_power[i] < 0 and trend_down and vol_ok:
                # Short: price below Alligator lips, bear power negative, 1w downtrend, volume spike
                signals[i] = -0.25
                position = -1
                entry_price = close_12h[i]
        else:
            # Update stoploss and check exits
            if position == 1:
                # Stoploss (2x ATR approximation)
                hl_range = pd.Series(high_12h - low_12h).rolling(window=10, min_periods=10).mean().values
                if close_12h[i] <= entry_price - 2.0 * hl_range[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: price crosses below Alligator teeth or bull power turns negative
                elif close_12h[i] < teeth[i] or bull_power[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Stoploss (2x ATR approximation)
                hl_range = pd.Series(high_12h - low_12h).rolling(window=10, min_periods=10).mean().values
                if close_12h[i] >= entry_price + 2.0 * hl_range[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: price crosses above Alligator teeth or bear power turns positive
                elif close_12h[i] > teeth[i] or bear_power[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals