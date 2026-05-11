#!/usr/bin/env python3
# 1d_Camarilla_R1_S1_Breakout_WeeklyTrend_Volume
# Hypothesis: Price breaking above/below R1/S1 Camarilla levels on 1d, filtered by 1w EMA50 trend and volume above 1.5x median. Exit on opposite Camarilla touch or ATR stop.
# Designed for 1d timeframe with 1w trend filter to reduce whipsaw in both bull and bear markets.
# Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_Camarilla_R1_S1_Breakout_WeeklyTrend_Volume"
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
    
    # 1d OHLCV
    close_1d = prices['close'].values
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    volume_1d = prices['volume'].values
    
    # --- 1w Trend Filter: EMA50 ---
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # --- 1d Camarilla Levels (based on previous day) ---
    # Calculate from previous 1d bar (shifted by 1 to avoid lookahead)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    # Camarilla R1 and S1 levels
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # --- Volume Filter: above 1.5x median of last 20 periods ---
    vol_median = pd.Series(volume_1d).rolling(window=20, min_periods=10).median().values
    vol_threshold = vol_median * 1.5
    
    # --- ATR for stoploss (14-period) ---
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
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
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_threshold[i]) or np.isnan(atr[i])):
            if position != 0:
                # Check stoploss
                if position == 1 and close_1d[i] <= entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_1d[i] >= entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine 1w trend
        trend_up = close_1d[i] > ema50_1w_aligned[i]
        trend_down = close_1d[i] < ema50_1w_aligned[i]
        
        # Volume filter: above 1.5x median
        vol_ok = volume_1d[i] > vol_threshold[i]
        
        if position == 0:
            # Look for entries only in direction of 1w trend with volume
            if close_1d[i] > camarilla_r1[i] and trend_up and vol_ok:
                # Long: price breaks above R1 + 1w uptrend + volume
                signals[i] = 0.25
                position = 1
                entry_price = close_1d[i]
            elif close_1d[i] < camarilla_s1[i] and trend_down and vol_ok:
                # Short: price breaks below S1 + 1w downtrend + volume
                signals[i] = -0.25
                position = -1
                entry_price = close_1d[i]
        else:
            # Update stoploss and check exits
            if position == 1:
                # Stoploss
                if close_1d[i] <= entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: price touches or crosses below S1
                elif close_1d[i] <= camarilla_s1[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Stoploss
                if close_1d[i] >= entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: price touches or crosses above R1
                elif close_1d[i] >= camarilla_r1[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals