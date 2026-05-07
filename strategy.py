#!/usr/bin/env python3
# 4h_ChaikinOscillator_VolumeTrend
# Hypothesis: 4-hour Chaikin Oscillator (MACD of A/D line) with volume confirmation and 1d EMA trend filter
# Chaikin Oscillator > 0 indicates buying pressure, < 0 indicates selling pressure
# Works in bull markets via positive Chaikin + breakout, in bear markets via negative Chaikin + breakdown
# Volume filter reduces false signals, trend filter avoids counter-trend trades
# Target: 20-50 trades per year (~80-200 over 4 years) with position size 0.25

name = "4h_ChaikinOscillator_VolumeTrend"
timeframe = "4h"
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
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Chaikin Oscillator = EMA(3, ADL) - EMA(10, ADL)
    # ADL = cumulative sum of ((close - low) - (high - close)) / (high - low) * volume
    clv = ((close - low) - (high - close)) / (high - low)
    clv = np.where((high - low) == 0, 0, clv)  # Avoid division by zero
    adl = np.cumsum(clv * volume)
    
    # EMA of ADL with periods 3 and 10
    adl_series = pd.Series(adl)
    ema3_adl = adl_series.ewm(span=3, adjust=False, min_periods=3).mean().values
    ema10_adl = adl_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    chaikin_osc = ema3_adl - ema10_adl
    
    # Volume ratio: current volume / 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 periods for volume MA and Chaikin stability
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(chaikin_osc[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Chaikin Oscillator signals
        chaikin_positive = chaikin_osc[i] > 0  # Buying pressure
        chaikin_negative = chaikin_osc[i] < 0  # Selling pressure
        
        # Volume confirmation: volume > 1.3x average
        volume_confirm = vol_ratio[i] > 1.3
        
        # Trend filter from 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: positive Chaikin + volume + uptrend
            if chaikin_positive and volume_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: negative Chaikin + volume + downtrend
            elif chaikin_negative and volume_confirm and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Chaikin turns negative or trend reversal
            if chaikin_osc[i] <= 0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Chaikin turns positive or trend reversal
            if chaikin_osc[i] >= 0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals