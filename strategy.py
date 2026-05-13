#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d trend filter (EMA34) and volume spike (2.0x MA20).
# Enters long when price breaks above Camarilla R3 with 1d bullish trend and volume > 2.0x MA20.
# Enters short when price breaks below Camarilla S3 with 1d bearish trend and volume > 2.0x MA20.
# Exits when price reaches Camarilla Pivot point (mean reversion to equilibrium).
# Uses discrete position sizing (0.30) to balance risk and reward.
# Designed for low trade frequency (~30-50/year) by requiring strict confluence.
# Camarilla levels provide institutional price structure; volume confirmation reduces false breakouts;
# 1d trend filter ensures alignment with higher timeframe direction, working in both bull and bear markets.

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume_v1"
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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for current 1d bar (based on previous 1d bar's OHLC)
    # Camarilla R3 = close + (high - low) * 1.1/2
    # Camarilla S3 = close - (high - low) * 1.1/2
    # Camarilla Pivot = (high + low + close) / 3
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3.0
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 2.0
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align Camarilla levels to 4h timeframe (wait for completed 1d bar)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get 1d data for trend filter (EMA34)
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 2.0x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(camarilla_pivot_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or \
           np.isnan(camarilla_s3_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3 with 1d bullish trend and volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.30
                position = 1
            # SHORT: Price breaks below Camarilla S3 with 1d bearish trend and volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches Camarilla Pivot (mean reversion)
            if close[i] >= camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price reaches Camarilla Pivot (mean reversion)
            if close[i] <= camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals