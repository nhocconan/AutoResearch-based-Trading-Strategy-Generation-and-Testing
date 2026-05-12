#!/usr/bin/env python3

# 6h_1D_MarketFacets_DynamicSupportResistance
# Hypothesis: Combines dynamic support/resistance from 6-period ATR-based channels with 1d market structure (Higher Highs/Lower Lows) to capture trend continuation while avoiding false breakouts in both bull and bear markets.
# In bull markets: buy when price pulls back to dynamic support in uptrend structure. In bear markets: sell when price rallies to dynamic resistance in downtrend structure.
# Uses volume confirmation to filter low-conviction moves. Targets 15-30 trades/year.

name = "6h_1D_MarketFacets_DynamicSupportResistance"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1d data for market structure
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)

    # Calculate 1d Higher Highs and Lower Lows for trend structure
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Higher High: current high > previous high
    higher_high = np.zeros(len(high_1d), dtype=bool)
    higher_high[1:] = high_1d[1:] > high_1d[:-1]
    
    # Lower Low: current low < previous low
    lower_low = np.zeros(len(low_1d), dtype=bool)
    lower_low[1:] = low_1d[1:] < low_1d[:-1]
    
    # Uptrend structure: HH and HL (Higher Low)
    higher_low = np.zeros(len(low_1d), dtype=bool)
    higher_low[1:] = low_1d[1:] > low_1d[:-1]
    uptrend_structure = higher_high & higher_low
    
    # Downtrend structure: LH and LL (Lower High)
    lower_high = np.zeros(len(high_1d), dtype=bool)
    lower_high[1:] = high_1d[1:] < high_1d[:-1]
    downtrend_structure = lower_high & lower_low
    
    # Align structures to 6t
    uptrend_aligned = align_htf_to_ltf(prices, df_1d, uptrend_structure.astype(int))
    downtrend_aligned = align_htf_to_ltf(prices, df_1d, downtrend_structure.astype(int))

    # Calculate 6-period ATR for dynamic channels
    atr_period = 6
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Dynamic support/resistance channels
    dynamic_support = close - (1.5 * atr)
    dynamic_resistance = close + (1.5 * atr)

    # Volume confirmation: current volume > 1.3x average of last 24 periods
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_ok = volume > (1.3 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(24, n):
        # Skip if any required data is NaN
        if (np.isnan(dynamic_support[i]) or np.isnan(dynamic_resistance[i]) or
            np.isnan(uptrend_aligned[i]) or np.isnan(downtrend_aligned[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend structure from 1d
        is_uptrend = uptrend_aligned[i] > 0.5
        is_downtrend = downtrend_aligned[i] > 0.5

        if position == 0:
            # LONG: Price at dynamic support in uptrend structure with volume
            if close[i] <= dynamic_support[i] and is_uptrend and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price at dynamic resistance in downtrend structure with volume
            elif close[i] >= dynamic_resistance[i] and is_downtrend and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches dynamic resistance or trend structure breaks
            if close[i] >= dynamic_resistance[i] or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches dynamic support or trend structure breaks
            if close[i] <= dynamic_support[i] or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals