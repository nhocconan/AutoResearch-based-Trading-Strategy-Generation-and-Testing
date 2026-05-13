#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation (>1.5x 20-bar avg) for entry timing.
# Uses 4h EMA50 for trend alignment (HTF), 1h Camarilla pivot levels for breakout entry, and volume confirmation to avoid false breakouts.
# Designed for 1h timeframe with low trade frequency (target 60-150 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following the 4h trend direction and requiring volume confirmation.
# Uses discrete position sizing (0.20) to control drawdown and reduce signal churn.

name = "1h_Camarilla_R1_S1_Breakout_4hEMA50_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA50 for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from previous 1h bar (LTF)
    # Camarilla: R1 = close + 1.0833*(high-low), S1 = close - 1.0833*(high-low)
    # We use previous bar's OHLC to avoid look-ahead
    camarilla_R1 = close + 1.0833 * (high - low)
    camarilla_S1 = close - 1.0833 * (high - low)
    
    # Shift by 1 to use previous bar's levels (no look-ahead)
    camarilla_R1 = np.roll(camarilla_R1, 1)
    camarilla_S1 = np.roll(camarilla_S1, 1)
    camarilla_R1[0] = np.nan
    camarilla_S1[0] = np.nan
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(camarilla_R1[i]) or 
            np.isnan(camarilla_S1[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R1, close > 4h EMA50, volume spike (>1.5x avg)
            if (high[i] > camarilla_R1[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below Camarilla S1, close < 4h EMA50, volume spike (>1.5x avg)
            elif (low[i] < camarilla_S1[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close position if price breaks below Camarilla S1
            if low[i] < camarilla_S1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Close position if price breaks above Camarilla R1
            if high[i] > camarilla_R1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals