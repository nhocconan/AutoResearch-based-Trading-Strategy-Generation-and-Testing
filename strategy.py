#!/usr/bin/env python3
"""
Hypothesis: 12h KAMA (Kaufman Adaptive Moving Average) with RSI and chop filter.
Long when KAMA is rising, RSI between 40-60, and chop > 61.8 (range market).
Short when KAMA is falling, RSI between 40-60, and chop > 61.8 (range market).
Exit when RSI crosses 50 or chop < 38.2 (trending market).
Designed for low trade frequency (12-37/year) with mean-reversion in range markets.
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
    
    # Load 1d data for chop calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d chop (choppiness index)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    atr_period = 14
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr = np.zeros_like(tr)
    for i in range(len(tr)):
        if i < atr_period:
            atr[i] = np.nan
        elif i == atr_period:
            atr[i] = np.nanmean(tr[1:i+1])
        else:
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Calculate highest high and lowest low over ATR period
    highest_high = np.zeros_like(high_1d)
    lowest_low = np.zeros_like(low_1d)
    for i in range(len(high_1d)):
        if i < atr_period:
            highest_high[i] = np.nan
            lowest_low[i] = np.nan
        else:
            highest_high[i] = np.nanmax(high_1d[i-atr_period+1:i+1])
            lowest_low[i] = np.nanmin(low_1d[i-atr_period+1:i+1])
    
    # Chop calculation
    sum_atr = np.zeros_like(atr)
    for i in range(len(sum_atr)):
        if i < atr_period:
            sum_atr[i] = np.nan
        else:
            sum_atr[i] = np.nansum(atr[i-atr_period+1:i+1])
    
    range_hl = highest_high - lowest_low
    chop = np.zeros_like(range_hl)
    for i in range(len(chop)):
        if range_hl[i] == 0 or np.isnan(sum_atr[i]) or np.isnan(range_hl[i]):
            chop[i] = np.nan
        else:
            chop[i] = 100 * np.log10(sum_atr[i] / range_hl[i]) / np.log10(atr_period)
    
    # Align chop to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 12h KAMA
    close_s = pd.Series(close)
    # Efficiency ratio
    change = np.abs(close - np.roll(close, 10))
    change[0] = np.nan
    volatility = np.abs(np.diff(close, prepend=np.nan))
    volatility_sum = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    er = change / volatility_sum
    er = np.nan_to_num(er, nan=0)
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    # KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Align KAMA to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate 12h RSI
    delta = np.diff(close, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rs = np.where(avg_loss == 0, 100, rs)
    rsi = 100 - (100 / (1 + rs))
    
    # Align RSI to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate 12h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after lookback
        # Skip if data not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(vol_avg_20[i])):
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
        
        # KAMA direction
        kama_rising = kama_aligned[i] > kama_aligned[i-1]
        kama_falling = kama_aligned[i] < kama_aligned[i-1]
        
        if position == 0:
            # Long: KAMA rising, RSI 40-60, chop > 61.8 (range market)
            if (kama_rising and 
                40 <= rsi_aligned[i] <= 60 and 
                chop_aligned[i] > 61.8 and
                volume[i] > 1.5 * vol_avg_20[i]):  # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling, RSI 40-60, chop > 61.8 (range market)
            elif (kama_falling and 
                  40 <= rsi_aligned[i] <= 60 and 
                  chop_aligned[i] > 61.8 and
                  volume[i] > 1.5 * vol_avg_20[i]):  # Volume confirmation
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: RSI crosses 50 or chop < 38.2 (trending market)
            exit_signal = False
            
            if position == 1:
                # Exit long: RSI crosses below 50 or chop < 38.2
                if (rsi_aligned[i] < 50 and rsi_aligned[i-1] >= 50) or chop_aligned[i] < 38.2:
                    exit_signal = True
            else:  # position == -1
                # Exit short: RSI crosses above 50 or chop < 38.2
                if (rsi_aligned[i] > 50 and rsi_aligned[i-1] <= 50) or chop_aligned[i] < 38.2:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_KAMA_RSI_Chop_Range"
timeframe = "12h"
leverage = 1.0
#%%