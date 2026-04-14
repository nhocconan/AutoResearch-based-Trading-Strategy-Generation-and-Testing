#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA13 on 1d for Bull/Bear Power
    ema13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = df_1d['high'].values - ema13_1d
    bear_power = df_1d['low'].values - ema13_1d
    
    # Align to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Calculate 6-period SMA of Bull/Bear Power for smoothing (using 6h data)
    # But we need to smooth on 1d then align, or smooth the aligned series
    # Smooth the aligned series with 6-period SMA (1 day = 4x6h bars)
    bull_power_smooth = pd.Series(bull_power_aligned).rolling(window=6, min_periods=6).mean().values
    bear_power_smooth = pd.Series(bear_power_aligned).rolling(window=6, min_periods=6).mean().values
    
    # Calculate ATR for volatility filter (6-period ATR on 6h)
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(13, 6)  # 13 for EMA, 6 for ATR and smoothing
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power_smooth[i]) or np.isnan(bear_power_smooth[i]) or 
            np.isnan(ema13_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND price above EMA13 (uptrend bias)
            # AND Bull Power expanding (current > previous) AND low volatility filter
            if (bull_power_smooth[i] > 0 and 
                price > ema13_aligned[i] and 
                bull_power_smooth[i] > bull_power_smooth[i-1] and
                atr[i] < np.nanmedian(atr[max(0, i-50):i+1]) * 1.5):  # volatility filter
                position = 1
                signals[i] = position_size
            # Short: Bear Power < 0 AND price below EMA13 (downtrend bias)
            # AND Bear Power expanding (more negative) AND low volatility filter
            elif (bear_power_smooth[i] < 0 and 
                  price < ema13_aligned[i] and 
                  bear_power_smooth[i] < bear_power_smooth[i-1] and
                  atr[i] < np.nanmedian(atr[max(0, i-50):i+1]) * 1.5):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bull Power turns negative OR price crosses below EMA13
            if bull_power_smooth[i] < 0 or price < ema13_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Bear Power turns positive OR price crosses above EMA13
            if bear_power_smooth[i] > 0 or price > ema13_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_ElderRay_PowerTrend"
timeframe = "6h"
leverage = 1.0