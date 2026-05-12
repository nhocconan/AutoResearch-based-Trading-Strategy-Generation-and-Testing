#/usr/bin/env python3
# 12h_Pivot_Point_Reversal_1dTrend_Volume
# Hypothesis: Daily pivot point reversals on 12h with 1d EMA trend filter and volume confirmation.
# Pivot points (PP, R1, S1) derived from prior day's OHLC act as institutional support/resistance.
# In ranging markets, price reverts to PP; in trending markets, breaks R1/S1 with volume signal continuation.
# The 1d EMA filter ensures alignment with higher timeframe trend to avoid counter-trend trades.
# Volume confirmation filters breakouts with low conviction. Designed for low-frequency, high-conviction trades.

name = "12h_Pivot_Point_Reversal_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1d Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 34-period EMA on 1d for trend direction (fibonacci-sensitive)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Daily Pivot Points (from prior 1d bar) ===
    # Use prior day's OHLC to avoid look-ahead
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Pivot Point (PP) = (H + L + C)/3
    pp = (prev_high + prev_low + prev_close) / 3.0
    # Resistance 1 (R1) = (2*PP) - L
    r1 = (2 * pp) - prev_low
    # Support 1 (S1) = (2*PP) - H
    s1 = (2 * pp) - prev_high
    
    # Align pivot levels to 12h timeframe (wait for 1d bar to close)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Volume Confirmation (20-period average on 12h) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend direction from 1d EMA
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Price crosses above S1 with volume and 1d uptrend (mean reversion from support)
            if (close[i] > s1_aligned[i] and vol_ok and trend_up):
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below R1 with volume and 1d downtrend (mean reversion from resistance)
            elif (close[i] < r1_aligned[i] and vol_ok and trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price crosses below PP or trend changes
            if (close[i] < pp_aligned[i] or not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above PP or trend changes
            if (close[i] > pp_aligned[i] or not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals