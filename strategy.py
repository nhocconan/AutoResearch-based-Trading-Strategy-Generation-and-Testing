#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend with daily volatility filter and volume confirmation.
# KAMA adapts to market noise - slow in ranging markets, fast in trending markets.
# Daily volatility filter (ATR ratio) avoids whipsaws in low volatility periods.
# Volume confirmation ensures institutional participation.
# Designed for low trade frequency (12-25/year) to minimize fee drag in 12h timeframe.
# Works in bull markets (KAMA rising with volume) and bear markets (KAMA falling with volume).
name = "12h_KAMA_VolatilityFilter_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for KAMA and volatility filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate KAMA using daily close (using previous day's data)
    # Efficiency Ratio (ER) = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, n=10))  # |close_t - close_{t-10}|
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=1)  # sum of |close_t - close_{t-1}|
    
    # Pad arrays to match length
    change_padded = np.concatenate([[np.nan]*10, change])
    volatility_padded = np.concatenate([[np.nan]*9, volatility, [np.nan]])  # sum over 10 periods needs 11 points
    
    # Calculate ER with proper alignment
    er = np.full_like(close_1d, np.nan)
    valid_idx = ~(np.isnan(change_padded) | np.isnan(volatility_padded))
    er[valid_idx] = change_padded[valid_idx] / volatility_padded[valid_idx]
    
    # Smoothing constants
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2  # where 0.6645 = 2/(2+1), 0.0645 = 2/(30+1)
    
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]  # start with first close
    for i in range(1, len(close_1d)):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate daily ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR using Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]):
                    result[i] = (result[i-1] * (period-1) + data[i]) / period
                else:
                    result[i] = np.nan
        return result
    
    atr = wilders_smoothing(tr, 14)
    atr_ma_50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    volatility_ratio = atr / atr_ma_50  # current ATR relative to 50-period average
    
    # Align KAMA and volatility ratio to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    volatility_ratio_aligned = align_htf_to_ltf(prices, df_1d, volatility_ratio)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_aligned[i]) or np.isnan(volatility_ratio_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        # Volatility filter: only trade when volatility is elevated (above average)
        vol_filter = volatility_ratio_aligned[i] > 1.0
        
        if position == 0:
            # Long: KAMA rising AND volume confirmation AND volatility filter
            kama_rising = kama_aligned[i] > kama_aligned[i-1]
            
            if vol_confirm and vol_filter and kama_rising:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling AND volume confirmation AND volatility filter
            elif vol_confirm and vol_filter and kama_aligned[i] < kama_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA falling OR volatility drops below average
            kama_falling = kama_aligned[i] < kama_aligned[i-1]
            low_volatility = volatility_ratio_aligned[i] < 0.8
            
            if kama_falling or low_volatility:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA rising OR volatility drops below average
            kama_rising = kama_aligned[i] > kama_aligned[i-1]
            low_volatility = volatility_ratio_aligned[i] < 0.8
            
            if kama_rising or low_volatility:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals