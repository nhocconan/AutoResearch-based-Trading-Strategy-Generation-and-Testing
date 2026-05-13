#!/usr/bin/env python3
# 12h_Camarilla_R2_S2_Pullback_1dTrend_Volume
# Hypothesis: Pullback to Camarilla R2/S2 levels on 12h with 1d trend filter and volume confirmation.
# Uses daily Camarilla levels calculated from previous day's high/low/close.
# Trend filter: 1d EMA50 (only trade in direction of higher timeframe trend).
# Volume confirmation: current volume > 1.5 x 20-period average.
# Entry: Buy near S2 in uptrend, Sell near R2 in downtrend (mean reversion within trend).
# Exit: Price crosses 1d EMA50 or reaches opposite Camarilla level (R3/S3).
# Designed to capture mean-reversion bounces within stronger trends, reducing false breakouts.
# Target: 15-25 trades/year per symbol to minimize fee drag while maintaining edge in bull/bear markets.

name = "12h_Camarilla_R2_S2_Pullback_1dTrend_Volume"
timeframe = "12h"
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

    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')

    # Calculate Camarilla levels for 12h using previous day's OHLC
    # R2 = C + (H-L)*1.1/4, S2 = C - (H-L)*1.1/4
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_r2 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s2 = prev_close - (prev_high - prev_low) * 1.1 / 4
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 2

    # Align Camarilla levels to 12h timeframe
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)

    # Trend filter: 1d EMA50
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(camarilla_r2_aligned[i]) or 
            np.isnan(camarilla_s2_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Pullback to S2 in uptrend with volume confirmation
            if (low[i] <= camarilla_s2_aligned[i] and 
                close[i] > camarilla_s2_aligned[i] and  # Confirm reversal
                close[i] > ema50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Pullback to R2 in downtrend with volume confirmation
            elif (high[i] >= camarilla_r2_aligned[i] and 
                  close[i] < camarilla_r2_aligned[i] and  # Confirm reversal
                  close[i] < ema50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses EMA50 down or reaches R3 (take profit)
            if close[i] < ema50_1d_aligned[i] or high[i] >= camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses EMA50 up or reaches S3 (take profit)
            if close[i] > ema50_1d_aligned[i] or low[i] <= camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals