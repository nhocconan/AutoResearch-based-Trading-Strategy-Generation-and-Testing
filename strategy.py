#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and 12h volume confirmation.
# Long when Alligator jaws (13-period SMMA) > teeth (8-period SMMA) > lips (5-period SMMA) with 1d close > EMA50 and 12h volume > 1.5x 20-period average.
# Short when jaws < teeth < lips with 1d close < EMA50 and 12h volume > 1.5x 20-period average.
# Exit on Alligator crossover (jaws crosses teeth). Uses discrete sizing (0.25) to minimize fee churn.
# Williams Alligator identifies strong trends, 1d EMA50 filters regime, volume confirms momentum.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

name = "12h_WilliamsAlligator_1dEMA50_12hVolumeConfirm"
timeframe = "12h"
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
    
    # --- 12h Indicators (LTF) ---
    # 12h volume confirmation: > 1.5x 20-period average (tight filter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_12h = volume > (1.5 * vol_ma_20)
    
    # Williams Alligator: SMMA(5), SMMA(8), SMMA(13) on median price
    median_price = (high + low) / 2.0
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value: SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev * (period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    lips = smma(median_price, 5)   # SMMA(5)
    teeth = smma(median_price, 8)  # SMMA(8)
    jaws = smma(median_price, 13)  # SMMA(13)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA(50)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaws[i]) or
            np.isnan(volume_confirm_12h[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Jaws > Teeth > Lips + 1d close > EMA50 + 12h volume confirmation
            if (jaws[i] > teeth[i] and teeth[i] > lips[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_confirm_12h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Jaws < Teeth < Lips + 1d close < EMA50 + 12h volume confirmation
            elif (jaws[i] < teeth[i] and teeth[i] < lips[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_confirm_12h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Jaws crosses below Teeth (Alligator sleeping)
            if jaws[i] < teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Jaws crosses above Teeth (Alligator sleeping)
            if jaws[i] > teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals