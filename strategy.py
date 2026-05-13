#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and 1d volume spike confirmation.
# Long when price breaks above Camarilla R1 AND close > 1d EMA34 AND 1d volume > 2.0 * 20-period average volume.
# Short when price breaks below Camarilla S1 AND close < 1d EMA34 AND 1d volume > 2.0 * 20-period average volume.
# Exit when price reverts to Camarilla Pivot point (mean reversion to equilibrium).
# Uses discrete position sizing (0.25) to limit fee churn. Designed for BTC/ETH robustness by trading institutional pivot levels with volume confirmation in trending markets.
# Target: 100-180 total trades over 4 years (25-45/year) for 4h timeframe.

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_1dVolumeSpike_v1"
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
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (2.0 * vol_ma_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Calculate Camarilla pivot levels from previous 1d bar (HTF)
    # Camarilla levels based on previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_open = df_1d['open'].shift(1).values
    
    # Typical price for Camarilla calculation
    typical_price = (prev_high + prev_low + prev_close) / 3.0
    
    # Camarilla R1, S1, Pivot levels
    camarilla_pivot = typical_price
    camarilla_r1 = camarilla_pivot + 1.1 * (prev_high - prev_low) / 12.0
    camarilla_s1 = camarilla_pivot - 1.1 * (prev_high - prev_low) / 12.0
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels for current day)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start after shift(1) warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or
            np.isnan(camarilla_pivot_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R1 AND price > 1d EMA34 AND volume spike
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike_aligned[i] > 0.5):  # True if volume spike aligned
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S1 AND price < 1d EMA34 AND volume spike
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to Camarilla Pivot (mean reversion)
            if close[i] <= camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reverts to Camarilla Pivot (mean reversion)
            if close[i] >= camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals