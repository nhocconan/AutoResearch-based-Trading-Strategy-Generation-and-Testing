#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d trend filter (EMA34) and volume spike confirmation.
# Long when price breaks above Camarilla R3 AND 1d EMA34 up AND 1d volume > 2.0 * 20-period average volume.
# Short when price breaks below Camarilla S3 AND 1d EMA34 down AND 1d volume > 2.0 * 20-period average volume.
# Exit when price returns to Camarilla Pivot level (mean reversion to equilibrium).
# Uses discrete position sizing (0.25) to limit fee churn. Designed for BTC/ETH robustness by capturing institutional breakouts with volume confirmation in trending markets.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
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
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d volume spike filter (HTF)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (2.0 * vol_ma_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Calculate Camarilla levels from prior 1d (HTF)
    # Camarilla levels use previous day's OHLC
    close_1d_prev = df_1d['close'].shift(1).values
    high_1d_prev = df_1d['high'].shift(1).values
    low_1d_prev = df_1d['low'].shift(1).values
    
    # Avoid index 0 issues
    close_1d_prev[0] = close_1d[0]
    high_1d_prev[0] = high_1d[0]
    low_1d_prev[0] = low_1d[0]
    
    pivot = (high_1d_prev + low_1d_prev + close_1d_prev) / 3.0
    range_1d = high_1d_prev - low_1d_prev
    r3 = pivot + (range_1d * 1.1 / 4.0)  # R3 = pivot + 1.1*(HL)/4
    s3 = pivot - (range_1d * 1.1 / 4.0)  # S3 = pivot - 1.1*(HL)/4
    
    # Align HTF levels to LTF (12h)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start after shift
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 AND 1d EMA34 up (bullish trend) AND volume spike
            if (close[i] > r3_aligned[i] and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and  # EMA34 rising
                volume_spike_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 AND 1d EMA34 down (bearish trend) AND volume spike
            elif (close[i] < s3_aligned[i] and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and  # EMA34 falling
                  volume_spike_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot (mean reversion)
            if close[i] <= pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to pivot (mean reversion)
            if close[i] >= pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals