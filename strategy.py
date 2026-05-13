#!/usr/bin/env python3
# 6h_PhaseAccumulation_CCI_Trend
# Hypothesis: Use Ehlers' Phase Accumulation cycle indicator (smoother than CCI) to identify
# momentum extremes and trend direction. Go long when phase advances above threshold
# with bullish CCI confirmation, short when phase declines below threshold with bearish CCI.
# Uses 1d trend filter to avoid counter-trend trades. Designed for low whipsaw in both bull/bear.

name = "6h_PhaseAccumulation_CCI_Trend"
timeframe = "6h"
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

    # Get 6h data for Phase Accumulation and CCI
    df_6h = get_htf_data(prices, '6h')
    
    # Phase Accumulation (Ehlers) - smooth momentum oscillator
    close_6h = df_6h['close'].values
    alpha = 0.07  # smoothing factor
    phase = np.zeros_like(close_6h)
    delta = np.zeros_like(close_6h)
    
    # Calculate phase accumulation
    for i in range(1, len(close_6h)):
        delta[i] = (1 - alpha) * (delta[i-1] + close_6h[i] - close_6h[i-1]) + alpha * (close_6h[i] - close_6h[i-1])
        phase[i] = phase[i-1] + delta[i]
    
    # CCI (20) on 6h
    typical_price = (df_6h['high'].values + df_6h['low'].values + df_6h['close'].values) / 3
    sma_tp = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = (typical_price - sma_tp) / (0.015 * mad)
    
    # Get 1d trend filter
    df_1d = get_htf_data(prices, '1d')
    sma50_1d = pd.Series(df_1d['close']).rolling(window=50, min_periods=50).mean().values
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)

    # Align 6h indicators
    phase_aligned = align_htf_to_ltf(prices, df_6h, phase)
    cci_aligned = align_htf_to_ltf(prices, df_6h, cci)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(phase_aligned[i]) or np.isnan(cci_aligned[i]) or 
            np.isnan(sma50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Phase momentum signals
        phase_curr = phase_aligned[i]
        phase_prev = phase_aligned[i-1] if i > 0 else phase_curr
        
        # Bullish: phase advancing (rising) above zero with CCI > 50
        phase_rising = phase_curr > phase_prev
        bullish_momentum = phase_rising and phase_curr > 0 and cci_aligned[i] > 50
        
        # Bearish: phase declining (falling) below zero with CCI < -50
        phase_falling = phase_curr < phase_prev
        bearish_momentum = phase_falling and phase_curr < 0 and cci_aligned[i] < -50
        
        # 1d trend filter
        price_above_sma50 = close[i] > sma50_1d_aligned[i]
        price_below_sma50 = close[i] < sma50_1d_aligned[i]

        if position == 0:
            # LONG: bullish momentum + price above 1d SMA50
            if bullish_momentum and price_above_sma50:
                signals[i] = 0.25
                position = 1
            # SHORT: bearish momentum + price below 1d SMA50
            elif bearish_momentum and price_below_sma50:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: bearish momentum OR price below 1d SMA50
            if bearish_momentum or not price_above_sma50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: bullish momentum OR price above 1d SMA50
            if bullish_momentum or not price_below_sma50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals