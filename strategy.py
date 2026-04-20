#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_KAMA_Direction_Choppiness_Filter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop (HTF for trend direction)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # === KAMA (Kaufman Adaptive Moving Average) on 12h close ===
    close_12h = df_12h['close'].values
    er = np.abs(close_12h[9:] - close_12h[:-9]) / np.sum(np.abs(np.diff(close_12h[:-1]), axis=0), axis=0) if len(close_12h) > 9 else np.array([])
    # Simplified ER calculation for speed and stability
    change = np.abs(np.diff(close_12h, 9))
    volatility = np.sum(np.abs(np.diff(close_12h)), axis=0) if len(close_12h) > 1 else np.array([])
    er_full = np.zeros_like(close_12h)
    if len(change) > 0 and len(volatility) > 0:
        er_full[9:] = np.where(volatility[8:] > 0, change / volatility[8:], 0)
    er_full[:9] = 0
    # Avoid division by zero in smoothing constants
    sc = (er_full * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    sc = np.where(er_full > 0, sc, (2/(30+1)) ** 2)  # fallback to fast EMA when ER=0
    kama_12h = np.zeros_like(close_12h)
    kama_12h[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        kama_12h[i] = kama_12h[i-1] + sc[i] * (close_12h[i] - kama_12h[i-1])
    # Alternative robust implementation using pandas for clarity
    close_series = pd.Series(close_12h)
    change = abs(close_series.diff(9))
    volatility = close_series.diff().abs().rolling(9).sum()
    er = change / volatility
    er = er.fillna(0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama_12h = close_series.ewm(alpha=sc, adjust=False).mean().values
    
    # Align KAMA to 4h
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # === Choppiness Index on 4h (range/trend filter) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    atr1 = high - low
    atr2 = np.abs(high - np.roll(close, 1))
    atr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(atr1, np.maximum(atr2, atr3))
    tr[0] = atr1[0]
    # True Range 14-period sum
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    range_hl = hh - ll
    chop = 100 * np.log10(tr_sum / np.where(range_hl > 0, range_hl, 1)) / np.log10(14)
    chop = np.where(range_hl > 0, chop, 50)  # default to neutral when range is zero
    
    # === 4h Volume Confirmation ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        kama_val = kama_12h_aligned[i]
        chop_val = chop[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(kama_val) or np.isnan(chop_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long when price > KAMA (uptrend), low chop (<40 = trending), volume confirmation
            if close_val > kama_val and chop_val < 40 and vol_ratio_val > 1.8:
                signals[i] = 0.25
                position = 1
            # Enter short when price < KAMA (downtrend), low chop, volume confirmation
            elif close_val < kama_val and chop_val < 40 and vol_ratio_val > 1.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below KAMA OR chop increases (range-bound)
            if close_val < kama_val or chop_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above KAMA OR chop increases
            if close_val > kama_val or chop_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals