#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for daily open (used as reference level)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily open price (from previous day)
    daily_open = df_1d['open'].values
    daily_open_aligned = align_htf_to_ltf(prices, df_1d, daily_open)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for volatility filter and position sizing
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(daily_open_aligned[i]) or np.isnan(vol_avg_20[i]) or 
            np.isnan(atr[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above daily open with volume surge and low volatility
            if (close[i] > daily_open_aligned[i] and 
                volume[i] > 1.8 * vol_avg_20[i] and
                atr[i] < 0.8 * np.median(atr[max(0, i-50):i+1])):  # Low volatility filter
                signals[i] = 0.20
                position = 1
            # Short: Price below daily open with volume surge and low volatility
            elif (close[i] < daily_open_aligned[i] and 
                  volume[i] > 1.8 * vol_avg_20[i] and
                  atr[i] < 0.8 * np.median(atr[max(0, i-50):i+1])):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: Price returns to daily open or volatility increases
            if position == 1:
                # Exit long: Price returns to daily open or volatility spikes
                if (close[i] <= daily_open_aligned[i] or 
                    atr[i] > 1.5 * np.median(atr[max(0, i-50):i+1])):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                # Exit short: Price returns to daily open or volatility spikes
                if (close[i] >= daily_open_aligned[i] or 
                    atr[i] > 1.5 * np.median(atr[max(0, i-50):i+1])):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1H_DailyOpen_Volume_Volatility_Filter"
timeframe = "1h"
leverage = 1.0