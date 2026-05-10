#!/usr/bin/env python3
# 4H_TRIX_VolumeSpike_12hTrend
# Hypothesis: TRIX (triple exponential moving average) momentum on 4h timeframe combined with volume spike and 12h EMA trend filter.
# TRIX captures momentum shifts; entering only on volume spikes avoids false signals. 12h EMA filter ensures we trade with higher timeframe trend.
# This reduces counter-trend trades in sideways or choppy markets, improving performance in both bull and bear regimes.
# Discrete position sizing (0.25) limits drawdown and controls trade frequency (target: 20-40 trades/year).
# The strategy avoids overtrading by requiring multiple confirmations: momentum, volume, and trend alignment.

name = "4H_TRIX_VolumeSpike_12hTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend direction
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate TRIX on 4h close: EMA(EMA(EMA(close, 12), 12), 12)
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = np.diff(ema3, prepend=ema3[0]) / ema3  # TRIX = (EMA3 - EMA3_prev) / EMA3_prev
    trix = np.where(ema3 != 0, trix, 0)
    
    # Volume filter: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Warmup for EMA and TRIX
    
    for i in range(start_idx, n):
        if np.isnan(ema_12h_aligned[i]) or np.isnan(trix[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 12h EMA50
        price_above_ema = close[i] > ema_12h_aligned[i]
        price_below_ema = close[i] < ema_12h_aligned[i]
        
        if position == 0:
            # Long entry: TRIX positive + above 12h EMA + volume spike
            if (trix[i] > 0 and 
                price_above_ema and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: TRIX negative + below 12h EMA + volume spike
            elif (trix[i] < 0 and 
                  price_below_ema and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX turns negative or volume drops below average
            if (trix[i] < 0 or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX turns positive or volume drops below average
            if (trix[i] > 0 or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals