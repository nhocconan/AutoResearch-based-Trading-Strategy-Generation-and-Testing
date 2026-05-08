#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13. Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 and Bear Power rising (less negative) with volume confirmation
# Short when Bear Power < 0 and Bull Power falling (less positive) with volume confirmation
# Uses 1d EMA34 for trend alignment to avoid counter-trend trades. Target: 15-25 trades/year.
name = "6h_ElderRay_BullBearPower_1dEMA34_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate EMA13 for Elder Ray (13-period EMA on close)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    # Rate of change of Bear Power (to detect bearish exhaustion)
    bear_power_roc = pd.Series(bear_power).diff(periods=3).values  # 3-bar ROC
    
    # Rate of change of Bull Power (to detect bullish exhaustion)
    bull_power_roc = pd.Series(bull_power).diff(periods=3).values  # 3-bar ROC
    
    # Volume confirmation - 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup period
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(bull_power_roc[i]) or np.isnan(bear_power_roc[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Bull Power positive AND Bear Power rising (less negative) with trend and volume
            if (bull_power[i] > 0 and 
                bear_power_roc[i] > 0 and 
                close[i] > ema34_1d_aligned[i] and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power negative AND Bull Power falling (less positive) with trend and volume
            elif (bear_power[i] < 0 and 
                  bull_power_roc[i] < 0 and 
                  close[i] < ema34_1d_aligned[i] and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power turns negative OR Bear Power accelerates downward
            if bull_power[i] <= 0 or bear_power_roc[i] < -0.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power turns positive OR Bull Power accelerates upward
            if bear_power[i] >= 0 or bull_power_roc[i] > 0.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals