#!/usr/bin/env python3
# Hypothesis: 6h Williams Alligator + 1d EMA50 trend filter + 6h volume confirmation.
# Long when Alligator jaws (13) > teeth (8) > lips (5) AND price > 1d EMA50 AND 6h volume > 2.0x 20-period average.
# Short when Alligator jaws < teeth < lips AND price < 1d EMA50 AND 6h volume > 2.0x 20-period average.
# Exit when Alligator lines re-cross (jaws < teeth for longs, jaws > teeth for shorts).
# Uses Alligator for trend identification, 1d EMA50 for higher-timeframe trend alignment, and volume confirmation to reduce false signals.
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag.

name = "6h_WilliamsAlligator_1dEMA50_6hVolumeConfirm"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 6h Indicators (LTF) ---
    # 6h volume confirmation: > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_6h = volume > (2.0 * vol_ma_20)
    
    # Williams Alligator (SMAs with specific periods and shifts)
    # Jaws: 13-period SMA, shifted 8 bars forward
    jaws = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA, shifted 5 bars forward
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA, shifted 3 bars forward
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) - trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data (due to shifts and min_periods)
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Jaws > Teeth > Lips (bullish alignment) AND price > 1d EMA50 AND volume spike
            if (jaws[i] > teeth[i] and teeth[i] > lips[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike_6h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Jaws < Teeth < Lips (bearish alignment) AND price < 1d EMA50 AND volume spike
            elif (jaws[i] < teeth[i] and teeth[i] < lips[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike_6h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Jaws < Teeth (Alligator lines re-cross - trend weakening)
            if jaws[i] < teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Jaws > Teeth (Alligator lines re-cross - trend weakening)
            if jaws[i] > teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals