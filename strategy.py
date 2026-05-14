#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation.
# Long when price > Alligator Jaw (teeth > lips) with 1d EMA50 > EMA200 (bull trend) and 12h volume > 1.5x 20-period average.
# Short when price < Alligator Jaw (teeth < lips) with 1d EMA50 < EMA200 (bear trend) and 12h volume > 1.5x 20-period average.
# Exit on opposite Alligator Jaw touch.
# Uses discrete position sizing (0.25) to minimize fee churn and volume confirmation to reduce false breakouts.
# Target: 50-150 total trades over 4 years = 12-37/year for 12h timeframe.
# Works in bull/bear: 1d EMA trend ensures strong trend alignment, Alligator provides trend-following structure.

name = "12h_WilliamsAlligator_1dEMATrend_12hVolumeConfirm"
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
    # 12h volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_12h = volume > (1.5 * vol_ma_20)
    
    # Williams Alligator (13,8,5 SMAs smoothed)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().rolling(window=8, min_periods=8).mean().values  # 13-period SMMA smoothed by 8
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().rolling(window=5, min_periods=5).mean().values   # 8-period SMMA smoothed by 5
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().rolling(window=3, min_periods=3).mean().values   # 5-period SMMA smoothed by 3
    
    # Alligator condition: Jaw > Teeth > Lips (bullish alignment) or Jaw < Teeth < Lips (bearish alignment)
    alligator_bull = (jaw > teeth) & (teeth > lips)
    alligator_bear = (jaw < teeth) & (teeth < lips)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA trend: EMA50 > EMA200 for bull, EMA50 < EMA200 for bear
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    ema_bull = ema_50_aligned > ema_200_aligned
    ema_bear = ema_50_aligned < ema_200_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(volume_confirm_12h[i]) or
            np.isnan(alligator_bull[i]) or
            np.isnan(alligator_bear[i]) or
            np.isnan(ema_bull[i]) or
            np.isnan(ema_bear[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price > Jaw + Alligator bullish + 1d EMA bull + volume confirmation
            if (close[i] > jaw[i] and 
                alligator_bull[i] and 
                ema_bull[i] and 
                volume_confirm_12h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price < Jaw + Alligator bearish + 1d EMA bear + volume confirmation
            elif (close[i] < jaw[i] and 
                  alligator_bear[i] and 
                  ema_bear[i] and 
                  volume_confirm_12h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price touches or crosses below Jaw
            if close[i] <= jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price touches or crosses above Jaw
            if close[i] >= jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals