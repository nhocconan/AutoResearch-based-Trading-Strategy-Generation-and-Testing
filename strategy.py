#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Breakout at Camarilla R3/S3 levels on 12h timeframe with 1d trend filter and volume confirmation.
# Long when price breaks above R3 with 1d uptrend and volume spike. Short when price breaks below S3 with 1d downtrend and volume spike.
# Uses volume spike (>1.5x 20-period average) to confirm breakouts and avoid false signals.
# Designed to work in both bull and bear markets by following the 1d trend direction.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
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

    # Get 12h data for price action and Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA50 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume average on 12h for confirmation
    volume_12h = df_12h['volume'].values
    vol_avg_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20)
    
    # Calculate Camarilla levels on 12h: R3, S3
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    camarilla_r3 = close_12h + 1.1 * (high_12h - low_12h) / 2
    camarilla_s3 = close_12h - 1.1 * (high_12h - low_12h) / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup period
        # Skip if any required value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Current 12h price
        price = close[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_spike = volume[i] > 1.5 * vol_avg_20_aligned[i]
        
        # Trend filter: price relative to 1d EMA50
        uptrend = price > ema50_1d_aligned[i]
        downtrend = price < ema50_1d_aligned[i]

        if position == 0:
            # LONG: Price breaks above R3 + volume spike + uptrend
            if price > camarilla_r3_aligned[i] and volume_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + volume spike + downtrend
            elif price < camarilla_s3_aligned[i] and volume_spike and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 OR volume dries up OR trend reverses
            if price < camarilla_s3_aligned[i] or not volume_spike or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 OR volume dries up OR trend reverses
            if price > camarilla_r3_aligned[i] or not volume_spike or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals