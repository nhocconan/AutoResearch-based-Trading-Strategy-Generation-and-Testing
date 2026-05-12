#!/usr/bin/env python3
# 4H_CAMARILLA_R1_S1_BREAKOUT_12HTREND_VOLUME_SPIKE
# Hypothesis: Buy when price breaks above Camarilla R1 with 12h uptrend and volume spike; sell when breaks below S1 with 12h downtrend and volume spike.
# Camarilla levels provide precise intraday support/resistance. Trend filter avoids counter-trend trades. Volume spike confirms breakout strength.
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue). Target: 20-40 trades/year.

name = "4H_CAMARILLA_R1_S1_BREAKOUT_12HTREND_VOLUME_SPIKE"
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
    
    # Calculate Camarilla levels from previous day
    # Use daily high, low, close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla calculations
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # 12h trend filter (EMA34)
    ema12_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema12_34_aligned = align_htf_to_ltf(prices, df_1d, ema12_34)
    
    # Volume spike detection (volume > 2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema12_34_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 with uptrend and volume spike
            if (close[i] > R1_aligned[i] and close[i] > ema12_34_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with downtrend and volume spike
            elif (close[i] < S1_aligned[i] and close[i] < ema12_34_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns below R1 or trend changes
            if close[i] < R1_aligned[i] or close[i] < ema12_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns above S1 or trend changes
            if close[i] > S1_aligned[i] or close[i] > ema12_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals