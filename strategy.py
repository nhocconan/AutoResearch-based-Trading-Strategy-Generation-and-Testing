#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Choppiness_Filtered_Keltner_Breakout"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    """
    12h Keltner Channel breakout with 1d Choppiness filter.
    - Long: Close > Upper Keltner Band (EMA20 + 2*ATR) and daily CHOP > 61.8 (ranging)
    - Short: Close < Lower Keltner Band (EMA20 - 2*ATR) and daily CHOP > 61.8 (ranging)
    - Exit: Opposite signal or price crosses EMA20
    - Uses 20-period EMA and ATR for bands
    - Target: 15-35 trades/year on 12h timeframe
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 20-period EMA for Keltner
    close_s = pd.Series(close)
    ema20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate ATR(20) for Keltner width
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Bands
    upper_keltner = ema20 + 2 * atr
    lower_keltner = ema20 - 2 * atr
    
    # Calculate Choppiness Index on daily
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    # True Range for 1d
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Highest High and Lowest Low over 14 periods
    highest_high = high_1d.rolling(window=14, min_periods=14).max().values
    lowest_low = low_1d.rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum(ATR_14) / (HH - LL)) / log10(14)
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr_14 / (highest_high - lowest_low)) / np.log10(14)
    chop[np.isnan(chop) | np.isinf(chop)] = 50  # default to middle when invalid
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if CHOP data not ready
        if np.isnan(chop_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade in ranging markets (CHOP > 61.8)
        if chop_aligned[i] > 61.8:
            if position == 0:
                # Long: Close above upper Keltner band
                if close[i] > upper_keltner[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Close below lower Keltner band
                elif close[i] < lower_keltner[i]:
                    signals[i] = -0.25
                    position = -1
            
            elif position == 1:
                # Exit long: Close crosses below EMA20 or opposite signal
                if close[i] < ema20[i] or close[i] < lower_keltner[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            
            elif position == -1:
                # Exit short: Close crosses above EMA20 or opposite signal
                if close[i] > ema20[i] or close[i] > upper_keltner[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # Trending market: stay flat or exit
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
    
    return signals