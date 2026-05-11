# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
4h_TurtleBreakout_1dTrend_Volume
Hypothesis: Classic Turtle Trading breakout on 4h timeframe (20-period high/low breakout),
filtered by 1d EMA50 trend and volume confirmation. Uses ATR-based position sizing and stoploss.
Designed for 20-50 trades/year per symbol to minimize fee drag while capturing strong trends.
Works in both bull and breakout markets by trading breakouts in direction of higher timeframe trend.
"""

name = "4h_TurtleBreakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
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
    
    # --- 4h Turtle Breakout: 20-period high/low ---
    # Highest high of last 20 periods
    high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Lowest low of last 20 periods
    low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # --- ATR for stoploss and position sizing ---
    tr1 = np.abs(high_4h - low_4h)
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr_4h = pd.Series(tr).rolling(window=20, min_periods=20).mean().values  # ATR(20)
    
    # --- Volume Filter: spike above 1.5x median of last 50 periods ---
    vol_median = pd.Series(volume_4h).rolling(window=50, min_periods=20).median().values
    vol_threshold = vol_median * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 50  # for EMA50, 20-period high/low, and ATR
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_threshold[i]) or np.isnan(atr_4h[i])):
            if position != 0:
                # Check stoploss
                if position == 1 and close_4h[i] <= entry_price - 2.0 * atr_4h[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_4h[i] >= entry_price + 2.0 * atr_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine 1d trend
        trend_up = close_4h[i] > ema50_1d_aligned[i]
        trend_down = close_4h[i] < ema50_1d_aligned[i]
        
        # Volume filter: spike above 1.5x median
        vol_ok = volume_4h[i] > vol_threshold[i]
        
        if position == 0:
            # Look for entries only in direction of 1d trend with volume spike
            if close_4h[i] > high_20[i] and trend_up and vol_ok:
                # Long: price breaks above 20-period high + 1d uptrend + volume spike
                signals[i] = 0.25
                position = 1
                entry_price = close_4h[i]
            elif close_4h[i] < low_20[i] and trend_down and vol_ok:
                # Short: price breaks below 20-period low + 1d downtrend + volume spike
                signals[i] = -0.25
                position = -1
                entry_price = close_4h[i]
        else:
            # Update stoploss and check exits
            if position == 1:
                # Stoploss
                if close_4h[i] <= entry_price - 2.0 * atr_4h[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: price returns to or below 20-period low (turtle exit)
                elif close_4h[i] <= low_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Stoploss
                if close_4h[i] >= entry_price + 2.0 * atr_4h[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: price returns to or above 20-period high (turtle exit)
                elif close_4h[i] >= high_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals