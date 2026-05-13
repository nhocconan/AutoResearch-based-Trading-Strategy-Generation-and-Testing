#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d trend filter (EMA34) and volume confirmation (1.5x MA20).
# Enters long when price breaks above Camarilla R3 level with 1d bullish trend and volume > 1.5x MA20.
# Enters short when price breaks below Camarilla S3 level with 1d bearish trend and volume > 1.5x MA20.
# Exits when price crosses the Camarilla pivot point (mean reversion).
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~12-37/year) by requiring strict confluence.
# Works in both bull and bear markets: 1d trend filter ensures alignment with higher timeframe direction,
# while Camarilla breakouts capture strong momentum moves from key intraday levels and volume confirmation reduces false signals.
# Camarilla levels are derived from the previous completed 1d bar to avoid look-ahead.

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume_v1"
timeframe = "12h"
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
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Calculate EMA(34) on 1d close
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous completed 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # Camarilla levels: based on previous day's range
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3
    camarilla_range = high_1d - low_1d
    camarilla_r3 = camarilla_pivot + camarilla_range * 1.1 / 4
    camarilla_s3 = camarilla_pivot - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (using previous completed 1d bar)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or \
           np.isnan(camarilla_pivot_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or \
           np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3 with 1d bullish trend and volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3 with 1d bearish trend and volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
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