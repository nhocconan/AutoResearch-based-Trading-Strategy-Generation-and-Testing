#!/usr/bin/env python3
"""
4h_1d_Camarilla_R3_S3_Breakout_Trend_Volume
Hypothesis: Breakouts at Camarilla R3/S3 levels on 4h, filtered by 1d EMA34 trend and volume above 50th percentile. Exit on opposite Camarilla level or ATR stop. Designed for low trade frequency (<50/year) to minimize fee drag and work in both bull/bear regimes via trend filter.
"""

name = "4h_1d_Camarilla_R3_S3_Breakout_Trend_Volume"
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
    
    # --- 1d Trend Filter: EMA34 ---
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # --- Camarilla Levels (based on previous day's OHLC) ---
    # Calculate daily OHLC from 1d data
    daily_open = df_1d['open'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Camarilla R3 and S3: H/L from previous day
    R3 = daily_close + (daily_high - daily_low) * 1.1 / 4
    S3 = daily_close - (daily_high - daily_low) * 1.1 / 4
    
    # Align to 4h: each 1d bar maps to 6*4h bars (since 24h/4h=6)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # --- Volume Filter: above median of last 50 periods ---
    vol_median = pd.Series(volume_4h).rolling(window=50, min_periods=20).median().values
    
    # --- ATR for stoploss (14-period) ---
    tr1 = np.abs(high_4h - low_4h)
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 50  # for EMA34, volume median, ATR
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(vol_median[i]) or np.isnan(atr[i])):
            if position != 0:
                # Check stoploss
                if position == 1 and close_4h[i] <= entry_price - 1.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_4h[i] >= entry_price + 1.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine 1d trend
        trend_up = close_4h[i] > ema34_1d_aligned[i]
        trend_down = close_4h[i] < ema34_1d_aligned[i]
        
        # Volume filter: above median
        vol_ok = volume_4h[i] > vol_median[i]
        
        if position == 0:
            # Look for entries only in direction of 1d trend with volume
            if close_4h[i] > R3_aligned[i] and trend_up and vol_ok:
                # Long: price breaks above R3 + 1d uptrend + volume
                signals[i] = 0.25
                position = 1
                entry_price = close_4h[i]
            elif close_4h[i] < S3_aligned[i] and trend_down and vol_ok:
                # Short: price breaks below S3 + 1d downtrend + volume
                signals[i] = -0.25
                position = -1
                entry_price = close_4h[i]
        else:
            # Update stoploss and check exits
            if position == 1:
                # Stoploss
                if close_4h[i] <= entry_price - 1.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: price touches or crosses below S3
                elif close_4h[i] < S3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Stoploss
                if close_4h[i] >= entry_price + 1.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: price touches or crosses above R3
                elif close_4h[i] > R3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals