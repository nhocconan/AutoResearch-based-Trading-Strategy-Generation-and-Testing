#!/usr/bin/env python3

"""
Hypothesis: 4-hour Volume-Weighted Average Price (VWAP) Reversion with 12-hour Exponential Moving Average trend filter.
Trades mean-reversion when price deviates significantly from VWAP, but only in the direction of the 12h EMA trend.
Uses Bollinger Bands to identify overextended conditions. Designed for low trade frequency (15-30 trades/year)
to minimize fee drag and work in both bull and bear markets by aligning with higher timeframe trend and using
statistical mean-reversion at extreme deviations.
"""

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
    
    # Load 12h data for trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # 12h EMA for trend filter (21-period)
    close_12h = df_12h['close'].values
    ema_21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # Calculate VWAP (typical price * volume) / volume
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    
    # Bollinger Bands (20, 2) around VWAP for mean reversion signals
    vwap_series = pd.Series(vwap)
    vwap_ma_20 = vwap_series.rolling(window=20, min_periods=20).mean().values
    vwap_std_20 = vwap_series.rolling(window=20, min_periods=20).std().values
    upper_band = vwap_ma_20 + 2.0 * vwap_std_20
    lower_band = vwap_ma_20 - 2.0 * vwap_std_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_21_12h_aligned[i]) or np.isnan(vwap_ma_20[i]) or 
            np.isnan(vwap_std_20[i]) or np.isnan(vwap[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price below lower Bollinger Band and uptrend bias (price > 12h EMA)
            if close[i] < lower_band[i] and close[i] > ema_21_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price above upper Bollinger Band and downtrend bias (price < 12h EMA)
            elif close[i] > upper_band[i] and close[i] < ema_21_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to VWAP (mean reversion complete) or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses above VWAP or closes below 12h EMA
                if close[i] > vwap[i] or close[i] < ema_21_12h_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses below VWAP or closes above 12h EMA
                if close[i] < vwap[i] or close[i] > ema_21_12h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_VWAP_MeanReversion_12hEMA21_Trend"
timeframe = "4h"
leverage = 1.0