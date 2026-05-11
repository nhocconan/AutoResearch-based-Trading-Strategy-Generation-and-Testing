#!/usr/bin/env python3
"""
4h_HTF_Camarilla_Pivot_Breakout_Trend
Hypothesis: In trending markets, price tends to respect Camarilla pivot levels (S1/S2/R1/R2) calculated from the prior 12h candle. A breakout above R1 with volume confirmation signals momentum continuation long; a breakdown below S1 signals continuation short. Uses 12h EMA50 as trend filter to avoid counter-trend trades. Designed for 4h timeframe with 12h HTF to reduce trade frequency and avoid fee drag. Works in both bull and bear markets by aligning trades with the higher timeframe trend.
"""

name = "4h_HTF_Camarilla_Pivot_Breakout_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for pivot calculation and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 12h Pivot Points (Camarilla) from previous 12h candle ---
    # Use prior 12h bar's OHLC (shifted by 1 to avoid look-ahead)
    prev_high = np.roll(df_12h['high'].values, 1)
    prev_low = np.roll(df_12h['low'].values, 1)
    prev_close = np.roll(df_12h['close'].values, 1)
    prev_high[0] = df_12h['high'].values[0]
    prev_low[0] = df_12h['low'].values[0]
    prev_close[0] = df_12h['close'].values[0]
    
    # Camarilla calculations
    range_ = prev_high - prev_low
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = pivot + (range_ * 1.1 / 12)
    r2 = pivot + (range_ * 1.1 / 6)
    s1 = pivot - (range_ * 1.1 / 12)
    s2 = pivot - (range_ * 1.1 / 6)
    
    # Align 12h levels to 4h timeframe
    pivot_4h = align_htf_to_ltf(prices, df_12h, pivot)
    r1_4h = align_htf_to_ltf(prices, df_12h, r1)
    r2_4h = align_htf_to_ltf(prices, df_12h, r2)
    s1_4h = align_htf_to_ltf(prices, df_12h, s1)
    s2_4h = align_htf_to_ltf(prices, df_12h, s2)
    
    # --- 12h EMA50 for trend filter ---
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # --- 4h Volume confirmation ---
    vol_avg_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 50  # for EMA50 and volume average
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema50_4h[i]) or np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(vol_avg_4h[i])):
            if position != 0:
                # Simple stop: exit if price crosses the 12h EMA50
                if position == 1 and close_4h[i] < ema50_4h[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_4h[i] > ema50_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Volume confirmation: current volume > 1.5x 4h average
        vol_confirm = volume_4h[i] > 1.5 * vol_avg_4h[i]
        
        if position == 0:
            # Look for breakout entries in direction of 12h trend
            if close_4h[i] > ema50_4h[i]:  # Uptrend
                if vol_confirm and close_4h[i] > r1_4h[i]:
                    signals[i] = 0.25  # long breakout above R1
                    position = 1
                    entry_price = close_4h[i]
            else:  # Downtrend
                if vol_confirm and close_4h[i] < s1_4h[i]:
                    signals[i] = -0.25  # short breakdown below S1
                    position = -1
                    entry_price = close_4h[i]
        else:
            # Manage existing position: exit on trend reversal or at opposite Camarilla level
            if position == 1:  # Long
                # Exit if trend turns down or price reaches S1 (contrarian level)
                if close_4h[i] < ema50_4h[i] or close_4h[i] < s1_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Short
                # Exit if trend turns up or price reaches R1 (contrarian level)
                if close_4h[i] > ema50_4h[i] or close_4h[i] > r1_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals