#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation (2.0x MA20).
# Enters long when price breaks above Camarilla R3 with 1w bullish trend (close > EMA34) and volume > 2.0x MA20.
# Enters short when price breaks below Camarilla S3 with 1w bearish trend (close < EMA34) and volume > 2.0x MA20.
# Exits when price reverts to the Camarilla pivot point (PP).
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~12-37/year) by requiring strict confluence: price breakout + HTF trend + volume spike.
# Camarilla levels provide high-probability reversal/breakout zones, while 1w EMA34 ensures alignment with weekly momentum.
# Volume threshold (2.0x) reduces false breakouts, improving signal quality in both bull and bear markets.

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_Volume_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Calculate EMA(34) on 1w close
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Get 1d data for Camarilla pivot levels (based on previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # R4 = Close + ((High-Low) * 1.5/2)
    # R3 = Close + ((High-Low) * 1.25/2)
    # R2 = Close + ((High-Low) * 1.1/2)
    # R1 = Close + ((High-Low) * 1.05/2)
    # PP = (High + Low + Close) / 3
    # S1 = Close - ((High-Low) * 1.05/2)
    # S2 = Close - ((High-Low) * 1.1/2)
    # S3 = Close - ((High-Low) * 1.25/2)
    # S4 = Close - ((High-Low) * 1.5/2)
    
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    camarilla_r3 = close_1d + ((high_1d - low_1d) * 1.25 / 2.0)
    camarilla_s3 = close_1d - ((high_1d - low_1d) * 1.25 / 2.0)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume filter: current volume > 2.0x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or \
           np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3 with 1w bullish trend and volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema34_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3 with 1w bearish trend and volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema34_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to Camarilla pivot point (PP)
            if close[i] < camarilla_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reverts to Camarilla pivot point (PP)
            if close[i] > camarilla_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals