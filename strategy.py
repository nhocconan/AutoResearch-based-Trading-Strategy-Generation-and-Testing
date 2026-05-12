#!/usr/bin/env python3
# 12h_1D_Keltner_Breakout_TrendVol
# Hypothesis: 12-hour breakouts from daily Keltner Channel upper/lower bands with daily EMA34 trend filter and volume spike confirmation.
# Keltner Channels adapt to volatility, providing dynamic support/resistance that works in both trending and ranging markets.
# Long when price breaks above upper band with volume spike and daily uptrend, short when breaks below lower band with volume spike and daily downtrend.
# Uses tight entry conditions to target 15-30 trades per year on 12h timeframe, avoiding overtrading.

name = "12h_1D_Keltner_Breakout_TrendVol"
timeframe = "12h"
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
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Daily data for Keltner Channels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily ATR for Keltner Channels (using 10-period ATR)
    atr_10_1d = pd.Series(np.maximum(
        df_1d['high'] - df_1d['low'],
        np.maximum(
            abs(df_1d['high'] - df_1d['close'].shift(1)),
            abs(df_1d['low'] - df_1d['close'].shift(1))
        )
    )).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner Channels: EMA20 ± 2*ATR
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema_20_1d + 2.0 * atr_10_1d
    lower_keltner = ema_20_1d - 2.0 * atr_10_1d
    
    # Align daily levels to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(upper_keltner_aligned[i]) or 
            np.isnan(lower_keltner_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper Keltner + volume spike + price above daily EMA34 (daily uptrend)
            if (close[i] > upper_keltner_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Keltner + volume spike + price below daily EMA34 (daily downtrend)
            elif (close[i] < lower_keltner_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters Keltner channel OR closes below daily EMA34
            if (close[i] < upper_keltner_aligned[i] and close[i] > lower_keltner_aligned[i]) or \
               close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters Keltner channel OR closes above daily EMA34
            if (close[i] < upper_keltner_aligned[i] and close[i] > lower_keltner_aligned[i]) or \
               close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals