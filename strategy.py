#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dTrend_Volume
Hypothesis: Price breaks out of Camarilla R3/S3 levels (calculated from previous day) on 4h timeframe, filtered by 1d EMA34 trend direction and volume spike (>1.5x median volume). Camarilla levels provide institutional support/resistance, trend filter ensures directional alignment, and volume confirms breakout strength. Designed for 20-35 trades/year to minimize fee decay while capturing strong directional moves in both bull and bear markets.
"""

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 1d Trend Filter: EMA34 ---
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # --- Camarilla Levels from Previous Day ---
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6, R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_r3 = np.zeros(len(df_1d))
    camarilla_s3 = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        h = high_1d[i]
        l = low_1d[i]
        c = close_1d[i]
        camarilla_r3[i] = c + (h - l) * 1.1 / 4
        camarilla_s3[i] = c - (h - l) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # --- Volume Filter: spike above 1.5x median of last 20 periods ---
    vol_median = pd.Series(volume_4h).rolling(window=20, min_periods=10).median().values
    vol_threshold = vol_median * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 20  # for EMA34 and volume median
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_threshold[i])):
            if position != 0:
                # Check stoploss (using ATR approximation from high-low)
                hl_range = np.abs(high_4h[i] - low_4h[i])
                if position == 1 and close_4h[i] <= entry_price - 1.5 * hl_range:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_4h[i] >= entry_price + 1.5 * hl_range:
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
            if close_4h[i] > camarilla_r3_aligned[i] and trend_up and vol_ok:
                # Long: price breaks above R3 + 1d uptrend + volume spike
                signals[i] = 0.25
                position = 1
                entry_price = close_4h[i]
            elif close_4h[i] < camarilla_s3_aligned[i] and trend_down and vol_ok:
                # Short: price breaks below S3 + 1d downtrend + volume spike
                signals[i] = -0.25
                position = -1
                entry_price = close_4h[i]
        else:
            # Update stoploss and check exits
            if position == 1:
                # Stoploss: 1.5x daily range from entry
                hl_range = np.abs(high_4h[i] - low_4h[i])
                if close_4h[i] <= entry_price - 1.5 * hl_range:
                    signals[i] = 0.0
                    position = 0
                # Exit: price returns to or below S3 level (mean reversion)
                elif close_4h[i] <= camarilla_s3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Stoploss: 1.5x daily range from entry
                hl_range = np.abs(high_4h[i] - low_4h[i])
                if close_4h[i] >= entry_price + 1.5 * hl_range:
                    signals[i] = 0.0
                    position = 0
                # Exit: price returns to or above R3 level (mean reversion)
                elif close_4h[i] >= camarilla_r3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals