#!/usr/bin/env python3
# 4H_WilliamsAlligator_ElderRay_Trend_Filter
# Hypothesis: Williams Alligator (13,8,5 SMAs) identifies trend direction and strength.
# Elder Ray (bull/bear power) confirms momentum behind the trend.
# Combined with volume confirmation (volume > 1.5x 20-period average) to filter false signals.
# Works in bull markets (Alligator aligned up, Elder Ray bullish) and bear markets (aligned down, bearish).
# Designed for low trade frequency (~20-30/year) with discrete sizing (0.25) to minimize fee drag.

name = "4H_WilliamsAlligator_ElderRay_Trend_Filter"
timeframe = "4h"
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
    
    # Williams Alligator: SMAs of median price (HL/2)
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # Blue line (13)
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values    # Red line (8)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values     # Green line (5)
    
    # Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    # Daily trend filter: EMA 50
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_threshold[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get daily close for trend determination
        close_1d_series = pd.Series(close_1d)
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d_series.values)
        
        is_uptrend = close_1d_aligned[i] > ema_50_1d_aligned[i]
        is_downtrend = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # Alligator aligned: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_short = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Long entry: Alligator aligned up + Bull Power positive + volume confirmation + daily uptrend
            if alligator_long and bull_power[i] > 0 and volume[i] > vol_threshold[i] and is_uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: Alligator aligned down + Bear Power negative + volume confirmation + daily downtrend
            elif alligator_short and bear_power[i] < 0 and volume[i] > vol_threshold[i] and is_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator alignment breaks or Bull Power turns negative
            if not alligator_long or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator alignment breaks or Bear Power turns positive
            if not alligator_short or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals