#!/usr/bin/env python3
"""
12h Williams Alligator + Volume Spike + 1d Trend Filter
Long when price > Alligator Jaw with bullish 1d trend (price > EMA34) and volume spike.
Short when price < Alligator Jaw with bearish 1d trend (price < EMA34) and volume filter.
Exit when price crosses Alligator Teeth.
Designed for low trade frequency (12-37/year) with strong trend capture.
"""
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
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Williams Alligator on 12h data
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    def smma(arr, period):
        """Smoothed Moving Average"""
        sma = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return sma
        sma[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            sma[i] = (sma[i-1] * (period-1) + arr[i]) / period
        return sma
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Calculate 12h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after lookback
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above Jaw with bullish 1d trend and volume spike
            if (close[i] > jaw[i] and 
                close[i] > ema34_aligned[i] and  # Bullish trend: price above EMA34
                volume[i] > 2.5 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = 0.30
                position = 1
            # Short: Price below Jaw with bearish 1d trend and volume filter
            elif (close[i] < jaw[i] and 
                  close[i] < ema34_aligned[i] and  # Bearish trend: price below EMA34
                  volume[i] > 1.5 * vol_avg_20[i]):  # Moderate volume
                signals[i] = -0.30
                position = -1
        else:
            # Exit conditions: price crosses Teeth
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below Teeth
                if close[i] < teeth[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above Teeth
                if close[i] > teeth[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "12H_WilliamsAlligator_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0