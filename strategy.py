#!/usr/bin/env python3
# 1d_WeeklySupportResistance_Breakout_Volume
# Hypothesis: Weekly support/resistance levels act as strong barriers in BTC/ETH.
# Price breaking above weekly resistance with volume surge indicates bullish momentum,
# while breaking below weekly support with volume surge indicates bearish momentum.
# Weekly timeframe reduces noise and captures major trend shifts. Volume confirmation
# filters false breakouts. Designed for low trade frequency (<25/year) to minimize fee drag.
# Works in bull markets (breakouts continue) and bear markets (breakdowns accelerate).

name = "1d_WeeklySupportResistance_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for support/resistance levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly support/resistance from previous week ---
    prev_week_high = df_1w['high'].values
    prev_week_low = df_1w['low'].values
    prev_week_close = df_1w['close'].values
    
    weekly_range = prev_week_high - prev_week_low
    
    # Weekly resistance (R1) and support (S1) - using classic pivot formula
    # R1 = 2*P - L, S1 = 2*P - H where P = (H+L+C)/3
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - prev_week_low
    weekly_s1 = 2 * weekly_pivot - prev_week_high
    
    # Align weekly levels to daily
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # --- Volume confirmation (2.0x 20-day average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # --- ATR for stoploss ---
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above weekly resistance with volume surge
            if close[i] > weekly_r1_aligned[i] and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly support with volume surge
            elif close[i] < weekly_s1_aligned[i] and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price drops below weekly support OR 2.5*ATR trailing stop
                if close[i] < weekly_s1_aligned[i] or close[i] < high[i] - 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises above weekly resistance OR 2.5*ATR trailing stop
                if close[i] > weekly_r1_aligned[i] or close[i] > low[i] + 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals