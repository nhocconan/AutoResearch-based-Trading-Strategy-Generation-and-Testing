#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above upper Donchian channel with volume > 2.0x average AND price > 1d EMA34.
# Short when price breaks below lower Donchian channel with volume > 2.0x average AND price < 1d EMA34.
# Exit on opposite Donchian level or trend reversal.
# Uses discrete position sizing (0.30) to balance profit potential and drawdown control.
# Target: 20-50 trades/year on 4h timeframe to minimize fee drag while capturing significant moves.
# Works in bull markets via breakout continuation and in bear markets via faded rallies at resistance.
# 4h timeframe is optimal for balancing trade frequency and signal quality based on proven patterns.

name = "4h_Donchian20_1dTrend_VolumeSpike_v1"
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
    
    # Get 4h data for Donchian channel calculation (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian(20) on 4h: upper = max(high,20), lower = min(low,20)
    high_series = pd.Series(high_4h)
    low_series = pd.Series(low_4h)
    upper_4h = high_series.rolling(window=20, min_periods=20).max().values
    lower_4h = low_series.rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for EMA34 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 2.0x 20-period average (on primary timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(upper_4h[i]) or np.isnan(lower_4h[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above upper Donchian with volume spike AND price > 1d EMA34
            if close[i] > upper_4h[i] and volume_filter[i] and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.30
                position = 1
            # SHORT: price breaks below lower Donchian with volume spike AND price < 1d EMA34
            elif close[i] < lower_4h[i] and volume_filter[i] and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below lower Donchian OR trend reversal (price < 1d EMA34)
            if close[i] < lower_4h[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: price breaks above upper Donchian OR trend reversal (price > 1d EMA34)
            if close[i] > upper_4h[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals