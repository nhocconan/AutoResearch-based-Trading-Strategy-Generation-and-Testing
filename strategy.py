#!/usr/bin/env python3
# 1d_Weekly_Keltner_Channel_Breakout
# Hypothesis: Price breaks above/below Keltner Channel (20, 2.0) derived from weekly timeframe with trend filter from weekly EMA50.
# Go long when price breaks above upper KC with weekly uptrend and volume confirmation.
# Go short when price breaks below lower KC with weekly downtrend and volume confirmation.
# Weekly Keltner Channel provides dynamic support/resistance that adapts to volatility, while weekly trend filter ensures alignment with higher timeframe momentum.
# Volume spike confirms institutional participation, reducing false breakouts.
# Works in bull markets (breakouts above upper KC in uptrend) and bear markets (breakdowns below lower KC in downtrend).
# Target: 10-25 trades/year per symbol to minimize fee drag.

name = "1d_Weekly_Keltner_Channel_Breakout"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for Keltner Channel calculation
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate Keltner Channel (20, 2.0) from weekly data
    # KC_Middle = EMA20 of close
    # KC_Upper = KC_Middle + 2.0 * ATR(20)
    # KC_Lower = KC_Middle - 2.0 * ATR(20)
    close_weekly = df_weekly['close'].values
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Calculate ATR(20)
    tr1 = high_weekly[1:] - low_weekly[1:]
    tr2 = np.abs(high_weekly[1:] - close_weekly[:-1])
    tr3 = np.abs(low_weekly[1:] - close_weekly[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # EMA20 for middle line
    ema20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Upper and lower bands
    kc_upper = ema20_weekly + 2.0 * atr_20
    kc_lower = ema20_weekly - 2.0 * atr_20
    
    # Weekly trend: EMA50
    ema50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly indicators to daily timeframe
    kc_upper_aligned = align_htf_to_ltf(prices, df_weekly, kc_upper)
    kc_lower_aligned = align_htf_to_ltf(prices, df_weekly, kc_lower)
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Volume spike: volume > 2.0 * 10-period average (approx 2 weeks worth)
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_spike = volume > 2.0 * vol_ma_10
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(kc_upper_aligned[i]) or 
            np.isnan(kc_lower_aligned[i]) or 
            np.isnan(ema50_weekly_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > KC_Upper + weekly uptrend + volume spike
            if close[i] > kc_upper_aligned[i] and close[i] > ema50_weekly_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < KC_Lower + weekly downtrend + volume spike
            elif close[i] < kc_lower_aligned[i] and close[i] < ema50_weekly_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below KC_Lower or trend reversal
            if close[i] < kc_lower_aligned[i] or close[i] < ema50_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above KC_Upper or trend reversal
            if close[i] > kc_upper_aligned[i] or close[i] > ema50_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals