# State-of-the-art 12h strategy based on proven patterns: Camarilla pivot + volume + regime
#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1wTrend_Volume
Hypothesis: Camarilla R1/S1 breakouts filtered by weekly EMA trend and volume spikes. 
Camarilla levels provide institutional-grade support/resistance that works in both bull and bear markets.
Weekly trend filter ensures alignment with long-term momentum, reducing whipsaws.
Volume spike confirms institutional participation. Target: 15-30 trades/year.
"""

name = "12h_Camarilla_R1S1_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    volume_12h = prices['volume'].values
    
    # --- Weekly Trend Filter: EMA34 ---
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # --- Daily Camarilla Levels (R1, S1) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1 = pivot + (range_1d * 1.0 / 12)
    s1 = pivot - (range_1d * 1.0 / 12)
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # --- Volume Filter: spike above 2.0x median of last 24 periods ---
    vol_median = pd.Series(volume_12h).rolling(window=24, min_periods=12).median().values
    vol_threshold = vol_median * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 34  # for EMA34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_threshold[i])):
            if position != 0:
                # Check stoploss
                atr_est = np.abs(high_12h[i] - low_12h[i])  # Simple range-based ATR estimate
                if position == 1 and close_12h[i] <= entry_price - 2.0 * atr_est:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_12h[i] >= entry_price + 2.0 * atr_est:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine weekly trend
        trend_up = close_12h[i] > ema34_1w_aligned[i]
        trend_down = close_12h[i] < ema34_1w_aligned[i]
        
        # Volume filter: spike above 2.0x median
        vol_ok = volume_12h[i] > vol_threshold[i]
        
        if position == 0:
            # Look for entries only in direction of weekly trend with volume spike
            if close_12h[i] > r1_aligned[i] and trend_up and vol_ok:
                # Long: price breaks above R1 + weekly uptrend + volume spike
                signals[i] = 0.25
                position = 1
                entry_price = close_12h[i]
            elif close_12h[i] < s1_aligned[i] and trend_down and vol_ok:
                # Short: price breaks below S1 + weekly downtrend + volume spike
                signals[i] = -0.25
                position = -1
                entry_price = close_12h[i]
        else:
            # Update stoploss and check exits
            atr_est = np.abs(high_12h[i] - low_12h[i])  # Simple range-based ATR estimate
            if position == 1:
                # Stoploss
                if close_12h[i] <= entry_price - 2.0 * atr_est:
                    signals[i] = 0.0
                    position = 0
                # Exit: price returns to or below S1 (mean reversion)
                elif close_12h[i] <= s1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Stoploss
                if close_12h[i] >= entry_price + 2.0 * atr_est:
                    signals[i] = 0.0
                    position = 0
                # Exit: price returns to or above R1 (mean reversion)
                elif close_12h[i] >= r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals