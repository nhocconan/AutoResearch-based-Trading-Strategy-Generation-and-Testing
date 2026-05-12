#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_RM
Hypothesis: Breakouts at daily Camarilla R1/S1 levels with volume confirmation and 1d trend filter.
Adds risk management via ATR trailing stop (3x ATR) to improve risk-adjusted returns.
Uses 4h timeframe to capture 20-30 trades/year. R1/S1 levels provide earlier entry than R3/S3.
1d trend filter avoids counter-trend trades. Volume confirmation ensures breakout strength.
Designed to work in both bull and bear regimes by following the 1d trend direction.
Trailing stop reduces drawdown during trend reversals.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_RM"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values

    # Calculate daily Camarilla levels (R1/S1 for earlier entry)
    camarilla_range = high_1d - low_1d
    r1 = close_1d + camarilla_range * 1.1 / 12
    s1 = close_1d - camarilla_range * 1.1 / 12

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: 1.5x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 1.5

    # ATR for risk management (3x ATR trailing stop)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # For long position tracking
    lowest_since_entry = 0.0   # For short position tracking

    for i in range(50, n):
        # Get aligned values for current 4h bar
        r1_aligned = align_htf_to_ltf(prices, df_1d, r1)[i]
        s1_aligned = align_htf_to_ltf(prices, df_1d, s1)[i]
        ema50_aligned = ema50_1d_aligned[i]
        vol_threshold_val = volume_threshold[i]
        atr_val = atr[i]

        # Skip if any required data is NaN
        if (np.isnan(r1_aligned) or np.isnan(s1_aligned) or 
            np.isnan(ema50_aligned) or np.isnan(vol_threshold_val) or np.isnan(atr_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price closes above daily R1 + volume spike (1.5x) + 1d uptrend
            if (close[i] > r1_aligned and
                volume[i] > vol_threshold_val and
                close[i] > ema50_aligned):
                signals[i] = 0.25
                position = 1
                highest_since_entry = close[i]
            # SHORT: Price closes below daily S1 + volume spike (1.5x) + 1d downtrend
            elif (close[i] < s1_aligned and
                  volume[i] > vol_threshold_val and
                  close[i] < ema50_aligned):
                signals[i] = -0.25
                position = -1
                lowest_since_entry = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Update highest price since entry
            if close[i] > highest_since_entry:
                highest_since_entry = close[i]
            
            # EXIT LONG: Price closes below daily S1 (reversal signal) OR trailing stop hit
            trailing_stop = highest_since_entry - 3.0 * atr_val
            if close[i] < s1_aligned or close[i] < trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest price since entry
            if close[i] < lowest_since_entry:
                lowest_since_entry = close[i]
            
            # EXIT SHORT: Price closes above daily R1 (reversal signal) OR trailing stop hit
            trailing_stop = lowest_since_entry + 3.0 * atr_val
            if close[i] > s1_aligned or close[i] > trailing_stop:  # Note: s1_aligned is lower bound, but we use r1 for short exit
                # Actually, for short we exit when price > r1 (resistance) OR trailing stop
                if close[i] > r1_aligned or close[i] > trailing_stop:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25

    return signals