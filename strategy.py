#!/usr/bin/env python3
# 12h_Camarilla_Pivot_1w_Trend_Volume_v1
# Hypothesis: Weekly trend direction from 1w EMA21 combined with daily Camarilla pivot levels (H3/L3) and volume confirmation
# provides high-probability mean-reversion entries in ranging markets and continuation in trends.
# The 12h timeframe reduces trade frequency to avoid fee drag while capturing multi-day moves.
# Works in bull/bear by using weekly trend filter to avoid counter-trend trades.

name = "12h_Camarilla_Pivot_1w_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly trend filter (1w EMA21)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    ema_21_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Daily Camarilla pivot levels (H3, L3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for pivot calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_ = prev_high - prev_low
    
    # Camarilla levels: H3/L3 are key reversal levels
    H3 = pivot + (range_ * 1.1 / 4)
    L3 = pivot - (range_ * 1.1 / 4)
    
    # Align to 12h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Volume confirmation: 12h volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Trend filter: price relative to weekly EMA21
    trend_up = close > ema_21_1w_aligned
    trend_down = close < ema_21_1w_aligned
    
    signals = np.zeros(n)
    
    # Start from sufficient lookback
    start_idx = max(20, 2)  # volume MA and pivot need previous day
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_21_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma[i]
        
        # Mean reversion at Camarilla H3/L3 with trend filter
        # In uptrend: buy near L3 (support), sell near H3 (resistance)
        # In downtrend: sell near H3 (resistance), buy near L3 (support)
        if vol_ok:
            # Long setup: price at L3 support in uptrend OR price at H3 resistance in downtrend (counter-trend bounce)
            if ((close[i] <= L3_aligned[i] * 1.002 and close[i] >= L3_aligned[i] * 0.998 and trend_up[i]) or
                (close[i] <= H3_aligned[i] * 1.002 and close[i] >= H3_aligned[i] * 0.998 and trend_down[i])):
                signals[i] = 0.25
            # Short setup: price at H3 resistance in uptrend OR price at L3 support in downtrend
            elif ((close[i] <= H3_aligned[i] * 1.002 and close[i] >= H3_aligned[i] * 0.998 and trend_up[i]) or
                  (close[i] <= L3_aligned[i] * 1.002 and close[i] >= L3_aligned[i] * 0.998 and trend_down[i])):
                signals[i] = -0.25
    
    return signals