#!/usr/bin/env python3
"""
4h_RVI_BullBear_1dTrend_50
Hypothesis: Relative Vigor Index (RVI) crossing above/below zero with 1d EMA50 trend filter and volume confirmation (1.5x 20-period median). RVI measures trend strength via price action relative to open-close range. In bull markets (price > 1d EMA50), long on RVI upward crosses; in bear markets (price < 1d EMA50), short on RVI downward crosses. Volume confirms momentum. Designed for fewer, higher-quality trades (target: 20-40/year) to avoid fee drag and work in both bull and bear regimes.
"""

name = "4h_RVI_BullBear_1dTrend_50"
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
    open_4h = prices['open'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 1d Trend Filter: EMA50 ---
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- RVI (10-period) ---
    numerator = close_4h - open_4h
    denominator = high_4h - low_4h
    # Avoid division by zero
    denom_safe = np.where(denominator == 0, 1e-10, denominator)
    price_change = numerator / denom_safe
    
    # Smooth numerator and denominator separately
    num_smooth = pd.Series(price_change).ewm(span=10, adjust=False, min_periods=10).mean().values
    den_smooth = pd.Series(np.ones_like(price_change)).ewm(span=10, adjust=False, min_periods=10).mean().values  # Always 1 for smoothed denominator
    rvi = num_smooth / den_smooth
    
    # --- Volume Filter: spike above 1.5x median of last 20 periods ---
    vol_median = pd.Series(volume_4h).rolling(window=20, min_periods=10).median().values
    vol_threshold = vol_median * 1.5
    
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
    start_idx = 50  # for EMA50 and RVI
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(rvi[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_threshold[i]) or np.isnan(atr[i])):
            if position != 0:
                # Check stoploss
                if position == 1 and close_4h[i] <= entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_4h[i] >= entry_price + 2.0 * atr[i]:
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
        
        # RVI signals: crossing zero
        rvi_cross_up = (rvi[i] > 0) and (rvi[i-1] <= 0)
        rvi_cross_down = (rvi[i] < 0) and (rvi[i-1] >= 0)
        
        if position == 0:
            # Look for entries only in direction of 1d trend with volume
            if rvi_cross_up and trend_up and vol_ok:
                # Long: RVI crosses up + 1d uptrend + volume spike
                signals[i] = 0.25
                position = 1
                entry_price = close_4h[i]
            elif rvi_cross_down and trend_down and vol_ok:
                # Short: RVI crosses down + 1d downtrend + volume spike
                signals[i] = -0.25
                position = -1
                entry_price = close_4h[i]
        else:
            # Update stoploss and check exits
            if position == 1:
                # Stoploss
                if close_4h[i] <= entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: RVI crosses back below zero
                elif rvi[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Stoploss
                if close_4h[i] >= entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: RVI crosses back above zero
                elif rvi[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals