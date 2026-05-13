#!/usr/bin/env python3
# Hypothesis: 12h KAMA trend alignment with 1d Bollinger Band mean reversion and volume spike confirmation.
# Long when price < 1d BB lower AND 12h KAMA rising AND 1d volume > 2.0 * 20-period average volume.
# Short when price > 1d BB upper AND 12h KAMA falling AND 1d volume > 2.0 * 20-period average volume.
# Exit when price crosses back inside Bollinger Bands or 12h KAMA reverses direction.
# Uses discrete position sizing (0.25) to limit fee churn. Designed for BTC/ETH robustness by capturing mean reversals in established trends with institutional volume confirmation.
# Target: 60-100 total trades over 4 years (15-25/year) for 12h timeframe.

name = "12h_KAMA_BollingerMeanReversion_1dVolumeSpike_v1"
timeframe = "12h"
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
    
    # Calculate 12h KAMA for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    # Calculate Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_12h, n=10))
    volatility = np.sum(np.abs(np.diff(close_12h, n=1)), axis=1)
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.full_like(close_12h, np.nan, dtype=float)
    kama[9] = close_12h[9]  # Seed
    for i in range(10, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    kama_12h = kama
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # Calculate 1d Bollinger Bands (20, 2.0) for mean reversion (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_lower = sma_20 - (2.0 * std_20)
    bb_upper = sma_20 + (2.0 * std_20)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    
    # Calculate 1d volume spike filter (HTF)
    df_1d_vol = get_htf_data(prices, '1d')
    volume_1d = df_1d_vol['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (2.0 * vol_ma_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d_vol, volume_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after BB/KAMA warmup
        # Skip if any required data is NaN
        if (np.isnan(kama_12h_aligned[i]) or 
            np.isnan(bb_lower_aligned[i]) or
            np.isnan(bb_upper_aligned[i]) or
            np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price below BB lower AND 12h KAMA rising AND volume spike
            if (close[i] < bb_lower_aligned[i] and 
                kama_12h_aligned[i] > kama_12h_aligned[i-1] and 
                volume_spike_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price above BB upper AND 12h KAMA falling AND volume spike
            elif (close[i] > bb_upper_aligned[i] and 
                  kama_12h_aligned[i] < kama_12h_aligned[i-1] and 
                  volume_spike_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back inside BB OR KAMA starts falling
            if (close[i] > bb_lower_aligned[i] or 
                kama_12h_aligned[i] < kama_12h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back inside BB OR KAMA starts rising
            if (close[i] < bb_upper_aligned[i] or 
                kama_12h_aligned[i] > kama_12h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals