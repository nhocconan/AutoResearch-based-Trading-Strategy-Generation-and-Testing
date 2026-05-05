#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w EMA trend filter and volume confirmation
# Williams Alligator: Jaw (EMA13, 8 periods smoothed), Teeth (EMA8, 5 periods smoothed), Lips (EMA5, 3 periods smoothed)
# Long when: Lips > Teeth > Jaw (bullish alignment) AND close > 1w EMA34 AND volume > 2.0x 20-period MA
# Short when: Lips < Teeth < Jaw (bearish alignment) AND close < 1w EMA34 AND volume > 2.0x 20-period MA
# Exit when: Alligator alignment breaks (Lips crosses Teeth or Teeth crosses Jaw)
# Uses Alligator for trend alignment, 1w EMA for higher timeframe trend, volume for conviction
# Timeframe: 12h, HTF: 1w. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_WilliamsAlligator_1wEMA34_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for EMA34
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # need sufficient data for EMA34
        return np.zeros(n)
    
    # Calculate 1w EMA34
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 12h SMMA (Smoothed Moving Average) for Williams Alligator
    # Jaw: SMMA13 (8 periods smoothed)
    if len(close) >= 13:
        sma_13 = pd.Series(close).rolling(window=13, min_periods=13).mean().values
        jaw = pd.Series(sma_13).ewm(span=8, adjust=False, min_periods=8).mean().values
    else:
        jaw = np.full(n, np.nan)
    
    # Teeth: SMMA8 (5 periods smoothed)
    if len(close) >= 8:
        sma_8 = pd.Series(close).rolling(window=8, min_periods=8).mean().values
        teeth = pd.Series(sma_8).ewm(span=5, adjust=False, min_periods=5).mean().values
    else:
        teeth = np.full(n, np.nan)
    
    # Lips: SMMA5 (3 periods smoothed)
    if len(close) >= 5:
        sma_5 = pd.Series(close).rolling(window=5, min_periods=5).mean().values
        lips = pd.Series(sma_5).ewm(span=3, adjust=False, min_periods=3).mean().values
    else:
        lips = np.full(n, np.nan)
    
    # Volume confirmation on 12h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Lips > Teeth > Jaw (bullish alignment) AND above 1w EMA34 AND volume spike
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Lips < Teeth < Jaw (bearish alignment) AND below 1w EMA34 AND volume spike
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks (Lips crosses below Teeth or Teeth crosses below Jaw)
            if lips[i] <= teeth[i] or teeth[i] <= jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks (Lips crosses above Teeth or Teeth crosses above Jaw)
            if lips[i] >= teeth[i] or teeth[i] >= jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals