# 6h_1w_Pivot_Momentum_Breakout
# Hypothesis: Use weekly pivot points (calculated from previous week's high/low/close) to establish key support/resistance levels.
# In trending markets (identified by 1-week EMA cross), breakouts above R1 or below S1 with volume confirmation indicate continuation.
# In ranging markets (price between R1 and S1), fade extreme touches of R2/S2 with mean reversion.
# This adapts to both bull and bear regimes by using weekly context and dynamic position sizing.
# Targets 15-25 trades/year per symbol by requiring weekly pivot alignment + volume + momentum filters.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-week data for pivot points and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week
    # Using (H+L+C)/3 formula for pivot, then support/resistance levels
    whigh = df_1w['high'].values
    loww = df_1w['low'].values
    closew = df_1w['close'].values
    
    # Pivot point = (H+L+C)/3
    pp = (whigh + loww + closew) / 3.0
    # Resistance 1 = (2*P) - L
    r1 = 2 * pp - loww
    # Support 1 = (2*P) - H
    s1 = 2 * pp - whigh
    # Resistance 2 = P + (H - L)
    r2 = pp + (whigh - loww)
    # Support 2 = P - (H - L)
    s2 = pp - (whigh - loww)
    
    # Align weekly pivot levels to 6h timeframe (using previous week's values)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Weekly trend: EMA(9) cross EMA(21) on weekly close
    closew_series = pd.Series(closew)
    ema9 = closew_series.ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21 = closew_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema9_aligned = align_htf_to_ltf(prices, df_1w, ema9)
    ema21_aligned = align_htf_to_ltf(prices, df_1w, ema21)
    # Trend = 1 if EMA9 > EMA21 (uptrend), -1 if EMA9 < EMA21 (downtrend)
    weekly_trend = np.where(ema9_aligned > ema21_aligned, 1, -1)
    
    # Volume confirmation: current volume > 1.5x 20-period median
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(weekly_trend[i]) or np.isnan(vol_threshold[i])):
            continue
        
        price = close[i]
        vol_ok = volume[i] > vol_threshold[i]
        
        # Determine market regime based on weekly trend
        is_uptrend = weekly_trend[i] == 1
        is_downtrend = weekly_trend[i] == -1
        
        # Position sizing: 0.25 for strong signals, 0.15 for weaker mean reversion
        strong_size = 0.25
        weak_size = 0.15
        
        # Trading logic
        signal = 0.0  # default to flat
        
        if is_uptrend:
            # In uptrend: look for long opportunities
            # Breakout above R1 with volume = continuation long
            if price > r1_aligned[i] and vol_ok:
                signal = strong_size
            # Mean reversion from S2 in uptrend = buy dip
            elif price < s2_aligned[i] and vol_ok:
                signal = weak_size
                
        elif is_downtrend:
            # In downtrend: look for short opportunities
            # Breakdown below S1 with volume = continuation short
            if price < s1_aligned[i] and vol_ok:
                signal = -strong_size
            # Mean reversion from R2 in downtrend = sell rally
            elif price > r2_aligned[i] and vol_ok:
                signal = -weak_size
        
        # In ranging markets (weak trend) or at extremes, consider mean reversion
        # Only if price is outside S2/R2 range
        elif abs(weekly_trend[i]) < 1:  # weak trend (shouldn't happen with our definition but safe)
            if price > r2_aligned[i] and vol_ok:
                signal = -weak_size  # sell at resistance
            elif price < s2_aligned[i] and vol_ok:
                signal = weak_size   # buy at support
        
        signals[i] = signal
    
    return signals

name = "6h_1w_Pivot_Momentum_Breakout"
timeframe = "6h"
leverage = 1.0