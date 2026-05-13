#!/usr/bin/env python3
# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA(34) trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 with price > 1d EMA34 (bullish trend) and volume > 2.0x 20-bar average.
# Short when price breaks below Camarilla S3 with price < 1d EMA34 (bearish trend) and volume > 2.0x average.
# Exit when price reverses and closes back inside the Camarilla H3/L3 levels.
# Uses discrete position sizing 0.25. Target: 50-150 total trades over 4 years on 6h timeframe.
# The Camarilla pivot levels provide strong intraday support/resistance, and R3/S3 breaks
# indicate institutional interest. Combined with 1d trend filter and volume confirmation,
# this should capture meaningful breakouts while avoiding false signals in choppy markets.
# Works in both bull and bear markets by trading with the higher timeframe trend.

name = "6h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_Volume_Confirm_v1"
timeframe = "6h"
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
    
    # Calculate Camarilla pivot levels (based on previous day's OHLC)
    # We need to calculate these for each 6h bar using the prior 1d bar's OHLC
    lookback = 20  # for volume average
    if n < lookback + 1:
        return np.zeros(n)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().values
    
    # Get 1d data for Camarilla pivot calculation and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # R4 = close + (high-low)*1.1/2
    # R3 = close + (high-low)*1.1/4
    # R2 = close + (high-low)*1.1/6
    # R1 = close + (high-low)*1.1/12
    # PP = (high+low+close)/3
    # S1 = close - (high-low)*1.1/12
    # S2 = close - (high-low)*1.1/6
    # S3 = close - (high-low)*1.1/4
    # S4 = close - (high-low)*1.1/2
    
    rng = high_1d - low_1d
    camarilla_r3 = close_1d + rng * 1.1 / 4
    camarilla_s3 = close_1d - rng * 1.1 / 4
    camarilla_h3 = close_1d + rng * 1.1 / 6  # for exit
    camarilla_l3 = close_1d - rng * 1.1 / 6  # for exit
    camarilla_pp = (high_1d + low_1d + close_1d) / 3  # pivot point
    
    # Calculate EMA(34) on 1d close
    if len(close_1d) < 34:
        ema_34_1d = np.full(len(close_1d), np.nan)
    else:
        ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 6h timeframe (wait for 1d bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback + 20, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3 with bullish 1d EMA trend and volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3 with bearish 1d EMA trend and volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 2.0 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes back below Camarilla H3 (reversal signal)
            if close[i] < camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes back above Camarilla L3 (reversal signal)
            if close[i] > camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals