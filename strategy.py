#!/usr/bin/env python3
# 6h_weekly_pivot_volume_breakout_v1
# Hypothesis: 6h breakouts above weekly pivot R1/S1 with volume confirmation and weekly trend filter.
# Weekly pivot levels act as dynamic support/resistance. Breakouts above R1 (long) or below S1 (short)
# with volume > 1.5x 20-period 6h average indicate institutional participation.
# Weekly EMA(34) determines primary trend to avoid counter-trend trades.
# Works in bull/bear: EMA filter avoids counter-trend trades, volume ensures momentum validity.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 12-37 trades/year on 6h.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_volume_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly HTF data for pivot calculation and EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Weekly OHLC for pivot points
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot: P = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Weekly R1 = 2*P - L, S1 = 2*P - H
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    
    # Weekly EMA(34) for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly indicators to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # 6h volume confirmation
    volume_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or
            np.isnan(s1_1w_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(volume_ma_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below weekly pivot OR weekly EMA turns bearish (price < EMA)
            if close[i] < pivot_1w_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above weekly pivot OR weekly EMA turns bullish (price > EMA)
            if close[i] > pivot_1w_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Volume confirmation: current volume > 1.5x 20-period average
            volume_confirmed = volume[i] > 1.5 * volume_ma_6h[i]
            
            if volume_confirmed:
                # Long entry: price breaks above weekly R1 AND above weekly EMA (uptrend)
                if close[i] > r1_1w_aligned[i] and close[i] > ema_34_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below weekly S1 AND below weekly EMA (downtrend)
                elif close[i] < s1_1w_aligned[i] and close[i] < ema_34_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals