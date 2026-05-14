#!/usr/bin/env python3
# 4h_TRIX_VolumeSpike_TrendFilter
# Hypothesis: TRIX momentum oscillator with 1d EMA trend filter and volume spike. TRIX captures smoothed momentum
# and turning points, effective in both trending and ranging markets when filtered by daily trend.
# Uses volume confirmation to avoid false signals. Targets 20-40 trades/year to minimize fee drag.
# TRIX(12) > 0 and rising = bullish momentum, TRIX(12) < 0 and falling = bearish momentum.

name = "4h_TRIX_VolumeSpike_TrendFilter"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # TRIX: Triple Exponential Moving Average (12-period)
    # Calculate EMA1, EMA2, EMA3 then % change
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = (ema3 / ema3.shift(1) - 1) * 100
    trix = trix.fillna(0).values  # First value will be NaN due to shift
    
    # Get daily EMA for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(12, 34, 20)  # Warmup for TRIX, daily EMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(trix[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # TRIX signals: rising TRIX = bullish momentum, falling TRIX = bearish momentum
        bullish_signal = trix[i] > 0 and trix[i] > trix[i-1]
        bearish_signal = trix[i] < 0 and trix[i] < trix[i-1]
        
        if position == 0:
            # Long entry: bullish TRIX + uptrend + volume spike
            if bullish_signal and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish TRIX + downtrend + volume spike
            elif bearish_signal and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX turns negative or trend reversal
            if trix[i] <= 0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX turns positive or trend reversal
            if trix[i] >= 0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals