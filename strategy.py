#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R1/S1 breakout with 1d trend filter (EMA34) and volume spike confirmation.
# Long when price breaks above R1 (Camarilla resistance 1) AND close > 1d EMA34 AND 1d volume > 2.0 * 20-period average volume.
# Short when price breaks below S1 (Camarilla support 1) AND close < 1d EMA34 AND 1d volume > 2.0 * 20-period average volume.
# Exit when price reverts to the 1d EMA34 level.
# Uses discrete position sizing (0.25) to limit fee churn. Designed for BTC/ETH robustness by trading mean reversals within the intraday Camarilla framework while filtering with higher timeframe trend and volume.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
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
    
    # Calculate Camarilla levels for each 1d bar (using prior day's OHLC)
    camarilla_R1 = np.full(len(df_1d), np.nan)
    camarilla_S1 = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        # Prior day's OHLC
        high_prior = df_1d['high'].iloc[i-1]
        low_prior = df_1d['low'].iloc[i-1]
        close_prior = df_1d['close'].iloc[i-1]
        # Camarilla R1 and S1
        camarilla_R1[i] = close_prior + 1.1 * (high_prior - low_prior) / 12
        camarilla_S1[i] = close_prior - 1.1 * (high_prior - low_prior) / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start after first bar to have prior day data
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or
            np.isnan(camarilla_R1_aligned[i]) or
            np.isnan(camarilla_S1_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above R1 AND close > 1d EMA34 AND volume spike
            if (close[i] > camarilla_R1_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S1 AND close < 1d EMA34 AND volume spike
            elif (close[i] < camarilla_S1_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price reverts to 1d EMA34 (mean reversion)
            if close[i] <= ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price reverts to 1d EMA34 (mean reversion)
            if close[i] >= ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals