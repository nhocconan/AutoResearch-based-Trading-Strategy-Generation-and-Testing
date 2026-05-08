#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyPivot_Breakout_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === Weekly pivot points ===
    prev_high_1w = np.roll(df_1w['high'].values, 1)
    prev_low_1w = np.roll(df_1w['low'].values, 1)
    prev_close_1w = np.roll(df_1w['close'].values, 1)
    prev_high_1w[0] = df_1w['high'].values[0]
    prev_low_1w[0] = df_1w['low'].values[0]
    prev_close_1w[0] = df_1w['close'].values[0]
    
    pivot_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    range_1w = prev_high_1w - prev_low_1w
    
    # Weekly R1 and S1 (key levels)
    r1_1w = pivot_1w + (range_1w * 1.1 / 12)
    s1_1w = pivot_1w - (range_1w * 1.1 / 12)
    r2_1w = pivot_1w + (range_1w * 1.1 / 6)
    s2_1w = pivot_1w - (range_1w * 1.1 / 6)
    
    # Align weekly levels to daily
    r1_1d = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1d = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1d = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1d = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # === Daily volume filter ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Trend filter: 50-day EMA ===
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_1d[i]) or np.isnan(s1_1d[i]) or np.isnan(r2_1d[i]) or np.isnan(s2_1d[i]) or
            np.isnan(ema50[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Breakout entry with volume confirmation
            long_cond = (close[i] > r2_1d[i] and volume[i] > vol_ma20[i])
            short_cond = (close[i] < s2_1d[i] and volume[i] > vol_ma20[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below S1 or trend reversal
            if close[i] < s1_1d[i] or close[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above R1 or trend reversal
            if close[i] > r1_1d[i] or close[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly pivot levels act as strong support/resistance due to institutional
# order flow. Breakouts above R2 or below S2 with volume indicate institutional
# participation and trend continuation. In ranging markets, price respects S1/R1.
# Works in bull markets via breakout continuation and in bear markets via
# mean reversion at key weekly levels. Uses 50-day EMA as trend filter and
# volume confirmation to avoid false breakouts. Targets 15-25 trades/year
# to minimize fee drag. Uses discrete sizing (0.25) to reduce churn. Effective
# on BTC/ETH as weekly pivot levels are widely watched by institutions.