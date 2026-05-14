#!/usr/bin/env python3
# Hypothesis: 1d Williams Alligator (Jaw=13, Teeth=8, Lips=5) with 1w EMA34 trend filter and volume confirmation (>1.5x 20-period average).
# Alligator lines: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3).
# Long when Lips > Teeth > Jaw (bullish alignment) AND close > 1w EMA34 (bullish HTF trend) AND volume > 1.5x MA20.
# Short when Lips < Teeth < Jaw (bearish alignment) AND close < 1w EMA34 (bearish HTF trend) AND volume > 1.5x MA20.
# Exit when Alligator alignment breaks (Lips crosses Teeth or Teeth crosses Jaw) OR price crosses 1w EMA34 in opposite direction.
# Uses 1w HTF for trend to reduce noise and overtrading. Volume confirmation reduces false signals.
# Target: 30-100 total trades over 4 years (7-25/year) to stay within fee drag limits for 1d timeframe.
# Williams Alligator identifies trending vs ranging markets; effective in both bull and bear markets when aligned with HTF trend.

name = "1d_WilliamsAlligator_1wEMA34_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(values, period):
    """Smoothed Moving Average (SMMA) aka Wilder's MA"""
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=float)
    result = np.empty_like(values, dtype=float)
    result[:] = np.nan
    # First value is simple average
    result[period-1] = np.mean(values[:period])
    # Subsequent values: (prev * (period-1) + current) / period
    for i in range(period, len(values)):
        result[i] = (result[i-1] * (period-1) + values[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1d Indicators (LTF) ---
    # Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3)
    jaw_raw = smma(close, 13)
    teeth_raw = smma(close, 8)
    lips_raw = smma(close, 5)
    
    # Shift as per Alligator definition
    jaw = np.roll(jaw_raw, 8)   # Jaw shifted 8 bars forward
    teeth = np.roll(teeth_raw, 5) # Teeth shifted 5 bars forward
    lips = np.roll(lips_raw, 3)   # Lips shifted 3 bars forward
    
    # 1d volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
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
            # LONG: Bullish alignment (Lips > Teeth > Jaw) AND close > 1w EMA34 AND volume confirm
            if (lips[i] > teeth[i] and 
                teeth[i] > jaw[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish alignment (Lips < Teeth < Jaw) AND close < 1w EMA34 AND volume confirm
            elif (lips[i] < teeth[i] and 
                  teeth[i] < jaw[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alignment breaks (Lips <= Teeth or Teeth <= Jaw) OR price < 1w EMA34 (trend change)
            if (lips[i] <= teeth[i] or 
                teeth[i] <= jaw[i] or 
                close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alignment breaks (Lips >= Teeth or Teeth >= Jaw) OR price > 1w EMA34 (trend change)
            if (lips[i] >= teeth[i] or 
                teeth[i] >= jaw[i] or 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals