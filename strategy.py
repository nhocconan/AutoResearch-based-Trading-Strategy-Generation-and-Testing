#!/usr/bin/env python3
"""
12h_VWAP_Deviation_Reversal_1dTrend
Hypothesis: Price deviations from daily VWAP with 1d trend filter and volume exhaustion signals 
mean-revert on 12h timeframe. Works in bull (dips in uptrend) and bear (bounces in downtrend). 
Targets 15-25 trades/year by requiring strong deviations and trend alignment.
"""
name = "12h_VWAP_Deviation_Reversal_1dTrend"
timeframe = "12h"
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
    
    # Get daily data for VWAP and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily VWAP: cumulative(volume * typical price) / cumulative(volume)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    vwap_numerator = (typical_price * df_1d['volume']).cumsum().values
    vwap_denominator = df_1d['volume'].cumsum().values
    # Avoid division by zero on first bar
    vwap = np.where(vwap_denominator != 0, vwap_numerator / vwap_denominator, typical_price.values)
    
    # Daily EMA trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily VWAP and EMA to 12h chart
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume exhaustion: current volume < 0.5x 20-period average (low volume on deviation)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_exhaustion = volume < (0.5 * vol_ma)
    
    # Price deviation from VWAP (%)
    deviation = (close - vwap_aligned) / vwap_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        if position == 0:
            # LONG: Price below VWAP (oversold) + uptrend + volume exhaustion
            if deviation[i] < -0.02 and close[i] > ema_1d_aligned[i] and volume_exhaustion[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price above VWAP (overbought) + downtrend + volume exhaustion
            elif deviation[i] > 0.02 and close[i] < ema_1d_aligned[i] and volume_exhaustion[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to VWAP or trend breaks
            if close[i] >= vwap_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to VWAP or trend breaks
            if close[i] <= vwap_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals