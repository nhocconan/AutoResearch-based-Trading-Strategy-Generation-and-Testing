#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_Volume
Hypothesis: Price breaks above Camarilla R1 or below S1 on 12h with 1d EMA34 trend filter and volume spike (1.5x median of last 20 bars).
Camarilla levels derived from prior day's range provide high-probability reversal/breakout levels. Trend filter ensures alignment with longer-term momentum.
Volume confirms conviction. Designed for 12-30 trades/year per symbol to minimize fee drag while capturing strong moves in both bull and bear markets.
"""

name = "12h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    volume_12h = prices['volume'].values
    
    # --- 1d Trend Filter: EMA34 ---
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # --- Camarilla Levels (from prior 1d bar) ---
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C, H, L are from previous day
    shift_close = np.roll(close_1d, 1)
    shift_high = np.roll(high_12h, 1)  # Wait, need 1d high/low
    # Actually need to use 1d high/low from df_1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    shift_high = np.roll(high_1d, 1)
    shift_low = np.roll(low_1d, 1)
    shift_close = np.roll(close_1d, 1)
    
    # Calculate Camarilla for each 1d bar (based on prior day)
    camarilla_R1 = shift_close + (shift_high - shift_low) * 1.1 / 12
    camarilla_S1 = shift_close - (shift_high - shift_low) * 1.1 / 12
    
    # Align to 12h timeframe (will be available after 1d bar closes)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # --- Volume Filter: spike above 1.5x median of last 20 periods ---
    vol_median = pd.Series(volume_12h).rolling(window=20, min_periods=10).median().values
    vol_threshold = vol_median * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period (need at least 2 days for Camarilla)
    start_idx = 1
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_threshold[i])):
            if position != 0:
                # Simple exit: reverse signal or flat
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for entries only in direction of 1d trend with volume spike
            if close_12h[i] > camarilla_R1_aligned[i] and close_12h[i] > ema34_1d_aligned[i] and volume_12h[i] > vol_threshold[i]:
                # Long: price breaks above R1 + 1d uptrend + volume spike
                signals[i] = 0.25
                position = 1
                entry_price = close_12h[i]
            elif close_12h[i] < camarilla_S1_aligned[i] and close_12h[i] < ema34_1d_aligned[i] and volume_12h[i] > vol_threshold[i]:
                # Short: price breaks below S1 + 1d downtrend + volume spike
                signals[i] = -0.25
                position = -1
                entry_price = close_12h[i]
        else:
            # Exit conditions: reverse signal or return to mean
            if position == 1:
                # Exit long: price breaks below S1 or returns to EMA34
                if close_12h[i] < camarilla_S1_aligned[i] or close_12h[i] < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above R1 or returns to EMA34
                if close_12h[i] > camarilla_R1_aligned[i] or close_12h[i] > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals