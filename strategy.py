#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_12hTrend_Volume
Hypothesis: Price breaks above Camarilla R1 (long) or below S1 (short) on 4h, filtered by 12h EMA50 trend and volume spike. Camarilla levels provide high-probability reversal/breakout points. Trend filter ensures alignment with longer-term momentum. Volume confirms conviction. Designed for 20-40 trades/year per symbol to minimize fee drag while capturing strong moves in both bull and bear markets.
"""

name = "4h_Camarilla_R1S1_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 12h Trend Filter: EMA50 ---
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # --- 4h Camarilla Pivot Levels (R1, S1) ---
    # Use previous day's high, low, close for Camarilla calculation
    # Since we're on 4h, we need daily OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot and Camarilla levels
    # For each 4h bar, use the most recent completed daily bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1_1d = pivot_1d + (range_1d * 1.0 / 12.0)
    s1_1d = pivot_1d - (range_1d * 1.0 / 12.0)
    
    # Align daily Camarilla levels to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # --- Volume Filter: spike above 1.5x median of last 30 periods ---
    vol_median = pd.Series(volume_4h).rolling(window=30, min_periods=10).median().values
    vol_threshold = vol_median * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 50  # for EMA50 and volume median
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_threshold[i])):
            if position != 0:
                # Check stoploss (2x ATR)
                # Calculate ATR(14) for stoploss
                if i >= 14:
                    tr1 = np.abs(high_4h[i] - low_4h[i])
                    tr2 = np.abs(high_4h[i] - close_4h[i-1])
                    tr3 = np.abs(low_4h[i] - close_4h[i-1])
                    tr = np.maximum(tr1, np.maximum(tr2, tr3))
                    # Simplified ATR calculation for stoploss
                    atr_14 = np.mean([
                        np.abs(high_4h[max(0, i-13):i+1] - low_4h[max(0, i-13):i+1]),
                        np.abs(high_4h[max(0, i-13):i+1] - np.roll(close_4h[max(0, i-13):i+1], 1)),
                        np.abs(low_4h[max(0, i-13):i+1] - np.roll(close_4h[max(0, i-13):i+1], 1))
                    ]) if i >= 13 else 0.0
                else:
                    atr_14 = 0.0
                
                if position == 1 and atr_14 > 0 and close_4h[i] <= entry_price - 2.0 * atr_14:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and atr_14 > 0 and close_4h[i] >= entry_price + 2.0 * atr_14:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine 12h trend
        trend_up = close_4h[i] > ema50_12h_aligned[i]
        trend_down = close_4h[i] < ema50_12h_aligned[i]
        
        # Volume filter: spike above 1.5x median
        vol_ok = volume_4h[i] > vol_threshold[i]
        
        if position == 0:
            # Look for entries only in direction of 12h trend with volume spike
            if close_4h[i] > r1_1d_aligned[i] and trend_up and vol_ok:
                # Long: price breaks above R1 + 12h uptrend + volume spike
                signals[i] = 0.25
                position = 1
                entry_price = close_4h[i]
            elif close_4h[i] < s1_1d_aligned[i] and trend_down and vol_ok:
                # Short: price breaks below S1 + 12h downtrend + volume spike
                signals[i] = -0.25
                position = -1
                entry_price = close_4h[i]
        else:
            # Update position and check exits
            if position == 1:
                # Exit: price returns to or below S1 (mean reversion)
                if close_4h[i] <= s1_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit: price returns to or above R1 (mean reversion)
                if close_4h[i] >= r1_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals