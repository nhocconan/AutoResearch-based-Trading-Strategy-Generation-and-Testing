#!/usr/bin/env python3
"""
6h_ADX_Alligator_WeeklyTrend_v1
Hypothesis: 6h ADX > 25 + Williams Alligator (Jaw/Teeth/Lips) alignment with weekly trend filter for strong trending markets.
Alligator provides dynamic support/resistance: Lips > Teeth > Jaw = bullish stack, reverse = bearish stack.
Weekly trend ensures we only trade in the direction of higher timeframe momentum.
Designed for 6h timeframe to capture medium-term trends with low frequency (target: 50-150 trades over 4 years).
Uses discrete position sizing (0.25) to minimize fee churn. Works in bull/bear via weekly trend alignment.
"""

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
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    # Weekly trend: 1 = uptrend (close > EMA50), -1 = downtrend (close < EMA50), 0 = invalid
    weekly_trend = np.where(ema_50_1w_aligned > 0, 
                            np.where(close > ema_50_1w_aligned, 1, -1), 
                            0)
    
    # Williams Alligator on 6h: SMAs of median price
    median_price = (high + low) / 2.0
    # Jaw: 13-period SMMA, shifted 8 bars
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # shift 8 bars forward
    jaw[:8] = np.nan
    # Teeth: 8-period SMMA, shifted 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # shift 5 bars forward
    teeth[:5] = np.nan
    # Lips: 5-period SMMA, shifted 3 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # shift 3 bars forward
    lips[:3] = np.nan
    
    # Alligator alignment: bullish when Lips > Teeth > Jaw, bearish when Lips < Teeth < Jaw
    alligator_bullish = (lips > teeth) & (teeth > jaw)
    alligator_bearish = (lips < teeth) & (teeth < jaw)
    
    # ADX (14) for trend strength
    # +DM, -DM, TR
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    up_move = np.concatenate([[np.nan], up_move])
    down_move = np.concatenate([down_move, [np.nan]])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar TR is just high-low
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    strong_trend = adx > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for weekly EMA, 14 for ADX, 13 for Alligator jaw)
    start_idx = max(50, 14, 13)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(adx[i]) or 
            np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(weekly_trend[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Entry conditions: Alligator alignment + ADX > 25 + weekly trend alignment
        if position == 0:
            # Long: Alligator bullish + strong trend + weekly uptrend
            if alligator_bullish[i] and strong_trend[i] and weekly_trend[i] == 1:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish + strong trend + weekly downtrend
            elif alligator_bearish[i] and strong_trend[i] and weekly_trend[i] == -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Alligator alignment breaks OR weekly trend turns down
            if not alligator_bullish[i] or weekly_trend[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Alligator alignment breaks OR weekly trend turns up
            if not alligator_bearish[i] or weekly_trend[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ADX_Alligator_WeeklyTrend_v1"
timeframe = "6h"
leverage = 1.0