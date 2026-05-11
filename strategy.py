#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_Volume
Hypothesis: Price breaks beyond Camarilla R1/S1 levels on daily timeframe, filtered by weekly trend (EMA34) and volume spike. Camarilla levels provide institutional support/resistance, weekly trend ensures alignment with higher timeframe momentum, and volume confirms breakout conviction. Designed for 7-25 trades/year per symbol to minimize fee drag while capturing strong moves.
"""

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "1d"
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
    
    # Daily OHLCV
    close_1d = prices['close'].values
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    volume_1d = prices['volume'].values
    
    # --- 1w Trend Filter: EMA34 ---
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # --- Daily Camarilla Pivot Levels (based on previous day) ---
    # Calculate pivot and levels from previous day's OHLC
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    
    # First day: use first available values
    prev_close[0] = close_1d[0]
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels
    R1 = pivot + 1.1 * range_val / 12.0
    S1 = pivot - 1.1 * range_val / 12.0
    
    # --- Volume Filter: spike above 1.5x median of last 30 days ---
    vol_median = pd.Series(volume_1d).rolling(window=30, min_periods=10).median().values
    vol_threshold = vol_median * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 30  # for volume median
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_threshold[i])):
            if position != 0:
                # Check stoploss (2% of price)
                if position == 1 and close_1d[i] <= entry_price * 0.98:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_1d[i] >= entry_price * 1.02:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine 1w trend
        trend_up = close_1d[i] > ema34_1w_aligned[i]
        trend_down = close_1d[i] < ema34_1w_aligned[i]
        
        # Volume filter: spike above 1.5x median
        vol_ok = volume_1d[i] > vol_threshold[i]
        
        if position == 0:
            # Look for entries only in direction of 1w trend with volume spike
            if close_1d[i] > R1[i] and trend_up and vol_ok:
                # Long: price breaks above R1 + 1w uptrend + volume spike
                signals[i] = 0.25
                position = 1
                entry_price = close_1d[i]
            elif close_1d[i] < S1[i] and trend_down and vol_ok:
                # Short: price breaks below S1 + 1w downtrend + volume spike
                signals[i] = -0.25
                position = -1
                entry_price = close_1d[i]
        else:
            # Update stoploss and check exits
            if position == 1:
                # Stoploss (2%)
                if close_1d[i] <= entry_price * 0.98:
                    signals[i] = 0.0
                    position = 0
                # Exit: price returns to or below pivot
                elif close_1d[i] <= pivot[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Stoploss (2%)
                if close_1d[i] >= entry_price * 1.02:
                    signals[i] = 0.0
                    position = 0
                # Exit: price returns to or above pivot
                elif close_1d[i] >= pivot[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals