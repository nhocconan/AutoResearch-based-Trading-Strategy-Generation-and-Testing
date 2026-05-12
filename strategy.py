#!/usr/bin/env python3
# 12H_CAMARILLA_R3_S3_BREAKOUT_1D_TREND_FILTER
# Hypothesis: Camarilla R3/S3 levels from daily timeframe act as strong support/resistance.
# In 1d uptrend (price > EMA34), go long when price breaks above R3 with volume confirmation.
# In 1d downtrend (price < EMA34), go short when price breaks below S3 with volume confirmation.
# Works in both bull and bear markets: trend filter avoids counter-trend trades, Camarilla breakouts capture momentum.
# Target: 15-25 trades/year on 12h timeframe.

name = "12H_CAMARILLA_R3_S3_BREAKOUT_1D_TREND_FILTER"
timeframe = "12h"
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
    
    # Camarilla R3 and S3 levels
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    r3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    s3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # EMA34 for trend filter
    ema34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: 1d uptrend + price breaks above R3 + volume spike
            if (close[i] > ema34_aligned[i] and 
                high[i] > r3_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: 1d downtrend + price breaks below S3 + volume spike
            elif (close[i] < ema34_aligned[i] and 
                  low[i] < s3_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal or price breaks below S3 (contrarian signal)
            if (close[i] <= ema34_aligned[i] or 
                low[i] < s3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal or price breaks above R3 (contrarian signal)
            if (close[i] >= ema34_aligned[i] or 
                high[i] > r3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals