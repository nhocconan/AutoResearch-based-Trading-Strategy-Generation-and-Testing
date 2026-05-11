#!/usr/bin/env python3
"""
4h_1d_WilliamsVixFix_Volume
Hypothesis: Williams Vix Fix (WVF) on 4h combined with 1d trend filter and volume confirmation.
- WVF measures market fear; high values indicate panic selling (potential bottom).
- Long when: WVF > 80 (extreme fear), 1d close > 1d EMA50 (uptrend), volume > 20-period average.
- Exit when: WVF < 20 (fear subsided) OR trend turns down.
- Designed to capture mean-reversion bounces in volatile markets (works in both bull and bear).
- Targets ~20-40 trades/year (80-160 over 4 years) to minimize fee drag.
"""

name = "4h_1d_WilliamsVixFix_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # --- Williams Vix Fix (WVF) on 4h ---
    # WVF = ((Highest High in period - Low) / Highest High in period) * 100
    # Highest High = rolling maximum of high over lookback period
    period = 22  # typical for WVF
    highest_high = pd.Series(high_4h).rolling(window=period, min_periods=period).max().values
    wvf = np.where(highest_high > 0, ((highest_high - low_4h) / highest_high) * 100, 0)
    
    # --- Volume Confirmation: 4h volume > 20-period average ---
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    # Start after warmup period
    start_idx = max(50, period)  # for EMA50 and WVF
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(wvf[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        trend_up = close_4h[i] > ema50_1d_aligned[i]
        
        # Volume confirmation
        vol_ok = volume_4h[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for long entries only in 1d uptrend with volume and extreme fear
            if wvf[i] > 80 and trend_up and vol_ok:
                # Long: extreme fear (WVF > 80) + uptrend + volume
                signals[i] = 0.25
                position = 1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: fear subsided (WVF < 20) OR trend turns down
                if wvf[i] < 20 or not trend_up:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
    
    return signals