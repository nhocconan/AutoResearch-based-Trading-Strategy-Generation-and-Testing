#!/usr/bin/env python3
# Hypothesis: 6h Bollinger Band Squeeze Breakout with 1w trend filter and 1d volume confirmation.
# Long when price breaks above upper BB(20,2) AND BBWidth < 20th percentile (squeeze) AND 1w close > 1w EMA50 (uptrend) AND 1d volume > 1.5 * 20-period average volume.
# Short when price breaks below lower BB(20,2) AND BBWidth < 20th percentile (squeeze) AND 1w close < 1w EMA50 (downtrend) AND 1d volume > 1.5 * 20-period average volume.
# Exit when price crosses back inside the Bollinger Bands (middle band).
# Uses discrete position sizing (0.25) to limit fee churn. Designed for BTC/ETH robustness by capturing low-volatility breakouts in trending markets with volume confirmation.
# Target: 60-100 total trades over 4 years (15-25/year) for 6h timeframe.

name = "6h_BollingerSqueezeBreakout_1wTrend_1dVolumeSpike_v1"
timeframe = "6h"
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
    
    # Calculate Bollinger Bands (20,2) on primary timeframe
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle
    
    # Calculate BBWidth percentile (20th) for squeeze condition
    bb_width_s = pd.Series(bb_width)
    bb_width_percentile_20 = bb_width_s.rolling(window=50, min_periods=50).quantile(0.20).values
    is_squeeze = bb_width < bb_width_percentile_20
    
    # Calculate 1w trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d volume spike filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * vol_ma_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after BB warmup
        # Skip if any required data is NaN
        if (np.isnan(bb_middle[i]) or 
            np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or
            np.isnan(is_squeeze[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above upper BB AND squeeze AND 1w uptrend AND volume spike
            if (close[i] > bb_upper[i] and 
                is_squeeze[i] and 
                close[i] > ema_50_1w_aligned[i] and  # Price above 1w EMA50 (uptrend)
                volume_spike_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below lower BB AND squeeze AND 1w downtrend AND volume spike
            elif (close[i] < bb_lower[i] and 
                  is_squeeze[i] and 
                  close[i] < ema_50_1w_aligned[i] and  # Price below 1w EMA50 (downtrend)
                  volume_spike_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back inside BB (below middle band)
            if close[i] < bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back inside BB (above middle band)
            if close[i] > bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals