# This is a placeholder for the actual solution code.
#!/usr/bin/env python3
"""
4h_1d_Weekly_Trend_Pullback
Hypothesis: Buy pullbacks to EMA21 in uptrends (1d EMA50 up + weekly EMA13 up) and sell rallies in downtrends (1d EMA50 down + weekly EMA13 down). Uses volume confirmation to avoid false signals.
Timeframe: 4h. Targets ~30 trades/year to minimize fee drag. Works in bull (trend-following) and bear (shorts in downtrends).
"""

name = "4h_1d_Weekly_Trend_Pullback"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d and weekly data for trend filters
    df_1d = get_htf_data(prices, '1d')
    df_w = get_htf_data(prices, '1w')
    if len(df_1d) < 2 or len(df_w) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 1d Trend Filter: EMA50 ---
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- Weekly Trend Filter: EMA13 ---
    close_w = df_w['close'].values
    ema13_w = pd.Series(close_w).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_w_aligned = align_htf_to_ltf(prices, df_w, ema13_w)
    
    # --- 4h EMA21 for pullback entries ---
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # --- Volume Confirmation: 4h volume > 20-period average ---
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period (max of EMA50, EMA13, EMA21, vol MA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(ema13_w_aligned[i]) or 
            np.isnan(ema21_4h[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend alignment: both 1d and weekly must agree
        bullish = ema50_1d_aligned[i] > ema13_w_aligned[i]  # 1d above weekly = bullish alignment
        bearish = ema50_1d_aligned[i] < ema13_w_aligned[i]  # 1d below weekly = bearish alignment
        
        # Volume confirmation
        vol_ok = volume_4h[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for long pullbacks in bullish alignment
            if bullish and vol_ok and close_4h[i] > ema21_4h[i] and low_4h[i] <= ema21_4h[i]:
                # Long: price pulls back to/touches EMA21 and bounces in bullish alignment
                signals[i] = 0.25
                position = 1
            # Look for short rallies in bearish alignment
            elif bearish and vol_ok and close_4h[i] < ema21_4h[i] and high_4h[i] >= ema21_4h[i]:
                # Short: price rallies to/touches EMA21 and rejects in bearish alignment
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: reverse signal or trend breakdown
            if position == 1:
                # Exit long: price breaks below EMA21 or trend turns bearish
                if close_4h[i] < ema21_4h[i] or not bullish:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above EMA21 or trend turns bullish
                if close_4h[i] > ema21_4h[i] or not bearish:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals