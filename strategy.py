#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and 1d volume spike confirmation.
# Long when price breaks above Camarilla R3 AND price > 1d EMA34 AND 1d volume > 2.0 * 20-period average volume.
# Short when price breaks below Camarilla S3 AND price < 1d EMA34 AND 1d volume > 2.0 * 20-period average volume.
# Exit when price crosses below Camarilla R3 (for longs) or above Camarilla S3 (for shorts).
# Uses discrete position sizing (0.30) to balance return and fee drag. Designed for BTC/ETH robustness by capturing institutional breakouts with volume confirmation in trending markets.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_1dVolumeSpike_v1"
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
    
    # Calculate Camarilla levels from previous 1d bar (HTF)
    close_prev_1d = df_1d['close'].shift(1).values
    high_prev_1d = df_1d['high'].shift(1).values
    low_prev_1d = df_1d['low'].shift(1).values
    
    camarilla_range = high_prev_1d - low_prev_1d
    camarilla_r3 = close_prev_1d + camarilla_range * 1.1 / 4
    camarilla_s3 = close_prev_1d - camarilla_range * 1.1 / 4
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start after 1d data shift
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3 AND price > 1d EMA34 AND volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike_aligned[i] > 0.5):  # True if volume spike aligned
                signals[i] = 0.30
                position = 1
            # SHORT: Price breaks below Camarilla S3 AND price < 1d EMA34 AND volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike_aligned[i] > 0.5):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Camarilla R3
            if close[i] < camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price crosses above Camarilla S3
            if close[i] > camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals