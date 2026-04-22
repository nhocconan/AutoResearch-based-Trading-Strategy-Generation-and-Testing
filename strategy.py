#!/usr/bin/env python3
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
    
    # Load daily data for Choppiness Index - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate Choppiness Index (14-period) on daily data
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # True Range
    tr1 = high_daily[1:] - low_daily[1:]
    tr2 = np.abs(high_daily[1:] - close_daily[:-1])
    tr3 = np.abs(low_daily[1:] - close_daily[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with original array
    
    # ATR (14-period)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of true ranges over 14 periods
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high_daily).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_daily).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = np.where((highest_high - lowest_low) != 0,
                    100 * np.log10(sum_tr / atr) / np.log10(14),
                    50)  # Default to neutral if no range
    
    # Align Choppiness Index to 4h timeframe with 2-bar delay for confirmation
    chop_aligned = align_htf_to_ltf(prices, df_daily, chop, additional_delay_bars=2)
    
    # Calculate 4-period RSI on 4h close prices
    delta = np.diff(close, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=4, min_periods=4).mean().values
    avg_loss = pd.Series(loss).rolling(window=4, min_periods=4).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Range-bound market filter: Choppiness Index > 61.8
        if chop_aligned[i] <= 61.8:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold (< 30) with volume confirmation
            if (rsi[i] < 30 and 
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (> 70) with volume confirmation
            elif (rsi[i] > 70 and 
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: RSI returns to neutral zone (40-60)
            if position == 1:
                if rsi[i] >= 40:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if rsi[i] <= 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4H_RSI_MeanReversion_ChopFilter_Volume"
timeframe = "4h"
leverage = 1.0