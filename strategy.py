#!/usr/bin/env python3
"""
6h_RVI_Trend_WeeklyBias
Hypothesis: Relative Vigor Index (RVI) confirms short-term trend on 6h, filtered by weekly trend (1w SMA) and monthly volatility regime. Long when RVI crosses above 0.5 with weekly uptrend and low volatility; short when RVI crosses below -0.5 with weekly downtrend and low volatility. Designed to capture trending moves while avoiding choppy periods. Target: 15-25 trades/year.
"""

name = "6h_RVI_Trend_WeeklyBias"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d and 1w data for filters
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 10 or len(df_1w) < 5:
        return np.zeros(n)
    
    # 6h OHLCV
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # --- RVI Calculation (10-period) ---
    numerator = (close_6h - open_6h) + 2 * (np.roll(close_6h, 1) - np.roll(open_6h, 1)) + \
                2 * (np.roll(close_6h, 2) - np.roll(open_6h, 2)) + (np.roll(close_6h, 3) - np.roll(open_6h, 3))
    denominator = (high_6h - low_6h) + 2 * (np.roll(high_6h, 1) - np.roll(low_6h, 1)) + \
                  2 * (np.roll(high_6h, 2) - np.roll(low_6h, 2)) + (np.roll(high_6h, 3) - np.roll(low_6h, 3))
    # Handle division by zero and first few values
    numerator[0:4] = 0
    denominator[0:4] = 1e-10  # small non-zero to avoid div/0
    rvi_raw = numerator / denominator
    rvi = pd.Series(rvi_raw).rolling(window=4, min_periods=4).mean().values
    
    # --- 1d Trend Filter: SMA50 ---
    close_1d = df_1d['close'].values
    sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    # --- 1w Trend Filter: SMA20 ---
    close_1w = df_1w['close'].values
    sma20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    sma20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma20_1w)
    
    # --- Monthly Volatility Regime Filter (using 1d data) ---
    # Calculate 20-day ATR as proxy for volatility
    tr1 = np.abs(df_1d['high'] - df_1d['low'])
    tr2 = np.abs(df_1d['high'] - np.roll(df_1d['close'], 1))
    tr3 = np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]
    atr20_1d = pd.Series(tr_1d).rolling(window=20, min_periods=20).mean().values
    # Normalize ATR by price to get volatility percentage
    vol_pct = atr20_1d / close_1d
    vol_pct_aligned = align_htf_to_ltf(prices, df_1d, vol_pct)
    # Low volatility regime: below 30th percentile of last 60 days
    vol_lookback = 60
    vol_percentile = pd.Series(vol_pct).rolling(window=vol_lookback, min_periods=20).quantile(0.30).values
    vol_percentile_aligned = align_htf_to_ltf(prices, df_1d, vol_percentile)
    vol_low = vol_pct_aligned < vol_percentile_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50  # for SMA50_1d and RVI
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(rvi[i]) or np.isnan(sma50_1d_aligned[i]) or 
            np.isnan(sma20_1w_aligned[i]) or np.isnan(vol_percentile_aligned[i])):
            if position != 0:
                # Simple exit: reverse signal or volatility regime change
                if position == 1 and (rvi[i] < 0 or not vol_low[i]):
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and (rvi[i] > 0 or not vol_low[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine conditions
        rvi_long_signal = rvi[i] > 0.5 and rvi[i-1] <= 0.5  # Cross above 0.5
        rvi_short_signal = rvi[i] < -0.5 and rvi[i-1] >= -0.5  # Cross below -0.5
        weekly_uptrend = close_6h[i] > sma20_1w_aligned[i]
        weekly_downtrend = close_6h[i] < sma20_1w_aligned[i]
        daily_uptrend = close_6h[i] > sma50_1d_aligned[i]
        daily_downtrend = close_6h[i] < sma50_1d_aligned[i]
        
        if position == 0:
            # Enter long: RVI long cross + weekly uptrend + low volatility
            if rvi_long_signal and weekly_uptrend and vol_low[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: RVI short cross + weekly downtrend + low volatility
            elif rvi_short_signal and weekly_downtrend and vol_low[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: RVI turns weak OR volatility increases OR weekly trend breaks
                if rvi[i] < 0 or not vol_low[i] or not weekly_uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: RVI turns weak OR volatility increases OR weekly trend breaks
                if rvi[i] > 0 or not vol_low[i] or not weekly_downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals