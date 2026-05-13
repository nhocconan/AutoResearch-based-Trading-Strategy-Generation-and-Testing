#!/usr/bin/env python3
# 4h_Camarilla_R3S3_Breakout_12hTrend_Volume
# Hypothesis: Breakout of Camarilla R3/S3 levels on 4h timeframe with 12h trend filter and volume confirmation.
# Long when price closes above R3 with bullish 12h trend and volume spike.
# Short when price closes below S3 with bearish 12h trend and volume spike.
# Designed to capture strong momentum moves while avoiding chop and false breakouts.

name = "4h_Camarilla_R3S3_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close arrays."""
    pivot = (high + low + close) / 3.0
    range_val = high - low
    R3 = close + range_val * 1.1 / 2.0
    S3 = close - range_val * 1.1 / 2.0
    return R3, S3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 4h data for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Camarilla levels on 4h data
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    R3_4h, S3_4h = calculate_camarilla(high_4h, low_4h, close_4h)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    # 12h EMA50 for trend direction
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike detection on 4h (volume > 1.5x 20-period average)
    vol_ma_4h = pd.Series(df_4h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_spike_threshold = vol_ma_4h * 1.5

    # Align all indicators to 4h timeframe
    R3_4h_aligned = align_htf_to_ltf(prices, df_4h, R3_4h)
    S3_4h_aligned = align_htf_to_ltf(prices, df_4h, S3_4h)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    vol_spike_aligned = align_htf_to_ltf(prices, df_4h, vol_spike_threshold)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(R3_4h_aligned[i]) or np.isnan(S3_4h_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_spike_aligned[i]) or
            np.isnan(volume[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Volume condition: current volume > spike threshold
        volume_spike = volume[i] > vol_spike_aligned[i]
        
        # Trend conditions
        bullish_trend = close[i] > ema50_12h_aligned[i]
        bearish_trend = close[i] < ema50_12h_aligned[i]
        
        # Breakout conditions
        breakout_long = close[i] > R3_4h_aligned[i]
        breakout_short = close[i] < S3_4h_aligned[i]

        if position == 0:
            # LONG: Breakout above R3 + bullish 12h trend + volume spike
            if breakout_long and bullish_trend and volume_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Breakout below S3 + bearish 12h trend + volume spike
            elif breakout_short and bearish_trend and volume_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below R3 or trend turns bearish
            if close[i] < R3_4h_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above S3 or trend turns bullish
            if close[i] > S3_4h_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals