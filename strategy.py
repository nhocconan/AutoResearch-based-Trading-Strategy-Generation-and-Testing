#!/usr/bin/env python3
# Hypothesis: 12h Williams %R extreme with 1d EMA(34) trend filter and 12h volume spike filter.
# Long when Williams %R < -80 (oversold) with 1d EMA bullish (close > EMA) and 12h volume > 2.0x 20-period average.
# Short when Williams %R > -20 (overbought) with 1d EMA bearish (close < EMA) and 12h volume > 2.0x 20-period average.
# Exit on Williams %R crossing above -50 for longs or below -50 for shorts.
# Uses discrete position sizing (0.25) to minimize fee churn and volume spike filter to reduce false signals.
# Williams %R captures momentum extremes, effective in both bull (buy dips) and bear (sell rallies) markets.
# 1d EMA ensures trend alignment, reducing counter-trend trades. Volume spike confirms institutional participation.

name = "12h_WilliamsR_Extreme_1dEMA34_12hVolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 12h Williams %R (14-period) ---
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = np.full(n, np.nan)
    for i in range(n):
        if highest_high[i] == lowest_low[i]:
            williams_r[i] = -50.0  # avoid division by zero
        else:
            williams_r[i] = -100 * (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i])
    
    # --- 12h volume spike: > 2.0x 20-period average ---
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_12h = volume > (2.0 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA(34)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(williams_r[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_spike_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold) + 1d EMA bullish + volume spike
            if (williams_r[i] < -80 and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike_12h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) + 1d EMA bearish + volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike_12h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses above -50 (momentum fading)
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses below -50 (momentum fading)
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals