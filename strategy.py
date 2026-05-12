#!/usr/bin/env python3
# 1D_WEEKLY_RANGE_BREAKOUT_VOLUME_CONFIRMATION
# Hypothesis: Weekly price range defines key support/resistance. Breakouts above weekly high or below weekly low with volume confirmation indicate institutional interest.
# In 1d uptrend (price > weekly VWAP), go long on weekly high breakout with volume spike; in downtrend, go short on weekly low breakout with volume spike.
# Weekly trend filter avoids counter-trend trades. Works in both bull and bear markets by following the weekly trend.
# Target: 10-20 trades/year on 1d timeframe.

name = "1D_WEEKLY_RANGE_BREAKOUT_VOLUME_CONFIRMATION"
timeframe = "1d"
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
    
    # Weekly data for range and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly high and low
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Weekly VWAP for trend filter
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    vwap = (typical_price * df_1w['volume']).cumsum() / df_1w['volume'].cumsum()
    weekly_vwap = vwap.values
    
    # Align to 1d timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    weekly_vwap_aligned = align_htf_to_ltf(prices, df_1w, weekly_vwap)
    
    # Volume spike detection (20-period volume average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure volume MA is stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(weekly_vwap_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Uptrend (price > weekly VWAP) + weekly high breakout + volume spike
            if (close[i] > weekly_vwap_aligned[i] and 
                high[i] > weekly_high_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend (price < weekly VWAP) + weekly low breakout + volume spike
            elif (close[i] < weekly_vwap_aligned[i] and 
                  low[i] < weekly_low_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal or price back below weekly high
            if (close[i] <= weekly_vwap_aligned[i] or 
                close[i] < weekly_high_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal or price back above weekly low
            if (close[i] >= weekly_vwap_aligned[i] or 
                close[i] > weekly_low_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals