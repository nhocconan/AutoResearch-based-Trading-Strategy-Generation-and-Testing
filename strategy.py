#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Designed to capture
# medium-term reversals at key pivot levels in both bull and bear markets by combining
# Camarilla structure, 1d trend filter, and volume strength. Targets 75-200 total trades over 4 years.

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
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
    
    # --- 4h Indicators (LTF) ---
    # Volume spike: > 2.0x 20-period average (strict threshold to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # Camarilla R1 and S1 levels from prior bar
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_r1 = close + 1.1 * (high - low) / 12
    camarilla_s1 = close - 1.1 * (high - low) / 12
    # Shift by 1 to use prior bar's levels (no look-ahead)
    camarilla_r1 = np.roll(camarilla_r1, 1)
    camarilla_s1 = np.roll(camarilla_s1, 1)
    camarilla_r1[0] = np.nan
    camarilla_s1[0] = np.nan
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) - trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(camarilla_r1[i]) or
            np.isnan(camarilla_s1[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R1 AND close > 1d EMA34 (bullish trend) AND volume spike
            if (close[i] > camarilla_r1[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S1 AND close < 1d EMA34 (bearish trend) AND volume spike
            elif (close[i] < camarilla_s1[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 1d EMA34 (trend change) OR touches Camarilla S1 (mean reversion)
            if close[i] < ema_34_1d_aligned[i] or close[i] < camarilla_s1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above 1d EMA34 (trend change) OR touches Camarilla R1 (mean reversion)
            if close[i] > ema_34_1d_aligned[i] or close[i] > camarilla_r1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals