#!/usr/bin/env python3
# 4H_CAMARILLA_R1S1_BREAKOUT_1D_TREND_FILTER
# Hypothesis: Camarilla R1/S1 levels act as strong support/resistance. In 1d uptrend (price > EMA34), go long when price breaks above R1 with volume confirmation. In 1d downtrend (price < EMA34), go short when price breaks below S1 with volume confirmation. Uses volume spike (1.5x average) to confirm breakout strength. Trend filter avoids counter-trend trades. Designed for 4h timeframe to balance trade frequency and performance.

name = "4H_CAMARILLA_R1S1_BREAKOUT_1D_TREND_FILTER"
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
    
    # Daily data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla levels
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels (R1, S1)
    R1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    S1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # EMA34 for trend filter
    ema34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume average (20-period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume spike condition (1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma_aligned[i]
        
        if position == 0:
            # LONG: 1d uptrend + price breaks above R1 + volume spike
            if (close[i] > ema34_aligned[i] and 
                high[i] > R1_aligned[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # SHORT: 1d downtrend + price breaks below S1 + volume spike
            elif (close[i] < ema34_aligned[i] and 
                  low[i] < S1_aligned[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 or trend reversal
            if (low[i] < S1_aligned[i] or 
                close[i] <= ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 or trend reversal
            if (high[i] > R1_aligned[i] or 
                close[i] >= ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals