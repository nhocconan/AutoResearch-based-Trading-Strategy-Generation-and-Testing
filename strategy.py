#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator with 1w EMA34 trend filter and volume confirmation (>1.8x 20-period average).
# Long when price > Alligator Jaw AND Jaw > Teeth AND Teeth > Lips (bullish alignment) AND close > 1w EMA34 AND volume confirm.
# Short when price < Alligator Jaw AND Jaw < Teeth AND Teeth < Lips (bearish alignment) AND close < 1w EMA34 AND volume confirm.
# Exit when Alligator alignment breaks (Jaw-Teeth-Lips no longer strictly ordered).
# Uses 1w HTF for primary trend to reduce noise and overtrading. Higher volume threshold reduces false signals.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
# Williams Alligator captures trending moves with built-in smoothing, reducing whipsaw in ranging markets.

name = "12h_WilliamsAlligator_1wEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 12h Indicators (LTF) ---
    # Williams Alligator: Smoothed Medians (Jaw=13, Teeth=8, Lips=5)
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward  
    # Lips: 5-period SMMA, shifted 3 bars forward
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_PRICE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift as per Alligator specification
    jaw = np.roll(jaw, 8)  # shifted 8 bars forward
    teeth = np.roll(teeth, 5)  # shifted 5 bars forward
    lips = np.roll(lips, 3)  # shifted 3 bars forward
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    # --- 1w Indicators (HTF) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # 1w EMA(34) - trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bullish Alligator alignment AND close > 1w EMA34 (bullish trend) AND volume confirm
            if (close[i] > jaw[i] and jaw[i] > teeth[i] and teeth[i] > lips[i] and
                close[i] > ema_34_1w_aligned[i] and
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish Alligator alignment AND close < 1w EMA34 (bearish trend) AND volume confirm
            elif (close[i] < jaw[i] and jaw[i] < teeth[i] and teeth[i] < lips[i] and
                  close[i] < ema_34_1w_aligned[i] and
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator alignment breaks (no longer bullish)
            if not (close[i] > jaw[i] and jaw[i] > teeth[i] and teeth[i] > lips[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator alignment breaks (no longer bearish)
            if not (close[i] < jaw[i] and jaw[i] < teeth[i] and teeth[i] < lips[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals