#!/usr/bin/env python3
# 1h_Camarilla_Pivot_Breakout_4hTrend_Volume
# Hypothesis: 1h Camarilla pivot breakouts with 4h EMA trend filter and volume confirmation.
# Uses daily pivots for mean reversion in range markets and breakouts in trends.
# Target: 15-35 trades/year (60-140 over 4 years).
# Works in bull (breakouts continue) and bear (mean reversion at extremes via trend filter).

name = "1h_Camarilla_Pivot_Breakout_4hTrend_Volume"
timeframe = "1h"
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
    
    # === 4h Trend Filter ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # 34-period EMA on 4h for trend direction
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # === Daily Camarilla Pivots (using previous day's range) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    range_ = prev_high - prev_low
    camarilla_h4 = prev_close + 1.1 * range_ / 2  # Resistance 4
    camarilla_l4 = prev_close - 1.1 * range_ / 2  # Support 4
    camarilla_h3 = prev_close + 1.1 * range_ / 4  # Resistance 3
    camarilla_l3 = prev_close - 1.1 * range_ / 4  # Support 3
    camarilla_h2 = prev_close + 1.1 * range_ / 6  # Resistance 2
    camarilla_l2 = prev_close - 1.1 * range_ / 6  # Support 2
    camarilla_h1 = prev_close + 1.1 * range_ / 12 # Resistance 1
    camarilla_l1 = prev_close - 1.1 * range_ / 12 # Support 1
    
    # Align to 1h (use previous day's levels for current day)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    
    # === Volume Confirmation (24-period average) ===
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend direction
        trend_up = close[i] > ema_34_4h_aligned[i]
        trend_down = close[i] < ema_34_4h_aligned[i]
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_24[i]
        
        if position == 0:
            # LONG: Price breaks above H3 with volume and uptrend
            if (close[i] > camarilla_h3_aligned[i] and vol_ok and trend_up):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below L3 with volume and downtrend
            elif (close[i] < camarilla_l3_aligned[i] and vol_ok and trend_down):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # EXIT LONG: Price returns to H4 or trend changes
            if (close[i] < camarilla_h4_aligned[i] or not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price returns to L4 or trend changes
            if (close[i] > camarilla_l4_aligned[i] or not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals