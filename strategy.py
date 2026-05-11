#!/usr/bin/env python3
"""
6h_Camarilla_R4_S4_Breakout_1wTrend_Volume
Hypothesis: Price breaking above/below R4/S4 Camarilla levels on 6h, filtered by 1w trend (price above/below 1w EMA50) and volume above 2x median.
Uses wider breakout levels (R4/S4) for fewer, higher-quality trades. Works in bull via uptrend breaks above R4, in bear via downtrend breaks below S4.
Volume confirms conviction. Target: 15-35 trades/year.
"""

name = "6h_Camarilla_R4_S4_Breakout_1wTrend_Volume"
timeframe = "6h"
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
    
    # 6h OHLCV
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    volume_6h = prices['volume'].values
    
    # --- 1w Trend Filter: EMA50 ---
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # --- 6h Camarilla Levels (based on previous day) ---
    # Calculate from previous 6h bar (shifted by 1 to avoid lookahead)
    prev_close = np.roll(close_6h, 1)
    prev_high = np.roll(high_6h, 1)
    prev_low = np.roll(low_6h, 1)
    prev_close[0] = close_6h[0]
    prev_high[0] = high_6h[0]
    prev_low[0] = low_6h[0]
    
    # Camarilla R4 and S4 levels (wider than R1/S1)
    camarilla_r4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # --- Volume Filter: above 2x median of last 20 periods ---
    vol_median = pd.Series(volume_6h).rolling(window=20, min_periods=10).median().values
    vol_threshold = vol_median * 2.0
    
    # --- ATR for stoploss (14-period) ---
    tr1 = np.abs(high_6h - low_6h)
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 50  # for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r4[i]) or np.isnan(camarilla_s4[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_threshold[i]) or np.isnan(atr[i])):
            if position != 0:
                # Check stoploss
                if position == 1 and close_6h[i] <= entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_6h[i] >= entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine 1w trend
        trend_up = close_6h[i] > ema50_1w_aligned[i]
        trend_down = close_6h[i] < ema50_1w_aligned[i]
        
        # Volume filter: above 2x median
        vol_ok = volume_6h[i] > vol_threshold[i]
        
        if position == 0:
            # Look for entries only in direction of 1w trend with volume
            if close_6h[i] > camarilla_r4[i] and trend_up and vol_ok:
                # Long: price breaks above R4 + 1w uptrend + volume
                signals[i] = 0.25
                position = 1
                entry_price = close_6h[i]
            elif close_6h[i] < camarilla_s4[i] and trend_down and vol_ok:
                # Short: price breaks below S4 + 1w downtrend + volume
                signals[i] = -0.25
                position = -1
                entry_price = close_6h[i]
        else:
            # Update stoploss and check exits
            if position == 1:
                # Stoploss
                if close_6h[i] <= entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: price touches or crosses below S4
                elif close_6h[i] <= camarilla_s4[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Stoploss
                if close_6h[i] >= entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: price touches or crosses above R4
                elif close_6h[i] >= camarilla_r4[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals