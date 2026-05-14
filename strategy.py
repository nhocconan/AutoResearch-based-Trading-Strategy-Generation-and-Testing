#!/usr/bin/env python3
# Hypothesis: 1d Williams Alligator (Jaw/Teeth/Lips) with 1w EMA34 trend filter and volume confirmation (>1.3x 20-period average).
# Long when Lips > Teeth > Jaw (bullish alignment) AND close > 1w EMA34 AND volume > 1.3x MA20.
# Short when Lips < Teeth < Jaw (bearish alignment) AND close < 1w EMA34 AND volume > 1.3x MA20.
# Exit when Alligator alignment breaks (Lips crosses Teeth or Teeth crosses Jaw).
# Uses 1w HTF for primary trend to reduce noise and overtrading. Volume confirmation reduces false signals.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
# Williams Alligator identifies trend phases; 1w EMA34 filters for higher-timeframe direction.

name = "1d_WilliamsAlligator_1wEMA34_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

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
    # Volume confirmation: > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma_20)
    
    # Williams Alligator: SMAs of median price (HL/2)
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values  # 13-period, shifted 8
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values   # 8-period, shifted 5
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values    # 5-period, shifted 3
    
    # --- 1w Indicators (HTF) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
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
            # LONG: Bullish Alligator alignment (Lips > Teeth > Jaw) AND close > 1w EMA34 AND volume confirm
            if (lips[i] > teeth[i] and 
                teeth[i] > jaw[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish Alligator alignment (Lips < Teeth < Jaw) AND close < 1w EMA34 AND volume confirm
            elif (lips[i] < teeth[i] and 
                  teeth[i] < jaw[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator alignment breaks (Lips <= Teeth or Teeth <= Jaw)
            if lips[i] <= teeth[i] or teeth[i] <= jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator alignment breaks (Lips >= Teeth or Teeth >= Jaw)
            if lips[i] >= teeth[i] or teeth[i] >= jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals