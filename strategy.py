#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation (2x MA20).
# Enters long when price breaks above Camarilla R3 with 1d bullish trend and volume > 2x MA20.
# Enters short when price breaks below Camarilla S3 with 1d bearish trend and volume > 2x MA20.
# Exits when price crosses the Camarilla pivot point (mean reversion).
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~20-50/year) by requiring strict confluence.
# Works in both bull and bear markets: 1d trend filter ensures alignment with higher timeframe direction,
# while Camarilla breakouts capture strong momentum moves and volume confirmation reduces false signals.
# Camarilla levels are derived from the previous completed 1d bar to avoid look-ahead.

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation and trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous completed 1d bar
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low), etc.
    # Pivot = (high + low + close)/3
    # We use the previous 1d bar's high, low, close to avoid look-ahead
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    # Set first value to NaN as there is no previous bar
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_pivot = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    camarilla_range = prev_high_1d - prev_low_1d
    camarilla_R3 = camarilla_pivot + 1.125 * camarilla_range
    camarilla_S3 = camarilla_pivot - 1.125 * camarilla_range
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar to complete)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Calculate EMA(34) on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 2x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or \
           np.isnan(camarilla_pivot_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or \
           np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3 with 1d bullish trend and volume spike
            if close[i] > camarilla_R3_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3 with 1d bearish trend and volume spike
            elif close[i] < camarilla_S3_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Camarilla pivot point (mean reversion)
            if close[i] < camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above Camarilla pivot point (mean reversion)
            if close[i] > camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals