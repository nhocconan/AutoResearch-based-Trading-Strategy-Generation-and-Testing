#!/usr/bin/env python3
# 1D_WeeklyPivot_Cross_TrendFollow
# Hypothesis: Uses weekly pivot points (PP, R1, S1) as key support/resistance levels on daily timeframe.
# Price crossing above/below weekly pivot with trend alignment (EMA50) and volume confirmation.
# Weekly pivot provides structure; daily EMA50 filters trend direction; volume confirms momentum.
# Designed for 1d timeframe to target 30-100 total trades over 4 years (7-25/year).
# Works in bull/bear by aligning with EMA50 trend and using pivot levels as dynamic S/R.

name = "1D_WeeklyPivot_Cross_TrendFollow"
timeframe = "1d"
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
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate ATR for volatility filter and stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate weekly pivot points from previous week's OHLC
    prev_weekly_close = df_1w['close'].shift(1).values
    prev_weekly_high = df_1w['high'].shift(1).values
    prev_weekly_low = df_1w['low'].shift(1).values
    
    # Standard pivot point: PP = (H + L + C) / 3
    pp = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    # R1 = 2*PP - L, S1 = 2*PP - H
    r1 = 2 * pp - prev_weekly_low
    s1 = 2 * pp - prev_weekly_high
    
    # Align weekly pivot levels to daily timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Get daily EMA for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume average for confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 14)  # Warmup for EMA50, volume MA, ATR
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema_50[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_50[i]
        downtrend = close[i] < ema_50[i]
        
        # Volume confirmation and volatility filter
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        volatility_filter = atr[i] > 0  # Ensure valid ATR
        
        if position == 0:
            # Long entry: price crosses above weekly R1 with volume confirmation, uptrend
            if close[i] > r1_aligned[i] and volume_confirm and uptrend and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: price crosses below weekly S1 with volume confirmation, downtrend
            elif close[i] < s1_aligned[i] and volume_confirm and downtrend and volatility_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls back below weekly PP or trend turns down
            if close[i] < pp_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises back above weekly PP or trend turns up
            if close[i] > pp_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals