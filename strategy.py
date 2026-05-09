#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_KAMA_Direction_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    """
    4h KAMA direction with 1d trend filter and volume confirmation.
    - KAMA(ER=10, FC=2, SC=30) adapts to volatility: faster in trending, slower in ranging.
    - Long: KAMA rising + close > KAMA + volume > 1.5x avg + price > 1d EMA(34)
    - Short: KAMA falling + close < KAMA + volume > 1.5x avg + price < 1d EMA(34)
    - Exit: Opposite KAMA direction or price crosses back through KAMA
    - Uses 1d EMA(34) for trend filter
    - Target: 20-40 trades/year on 4h timeframe
    """
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # KAMA calculation on 4h close
    close_series = pd.Series(close)
    # Efficiency Ratio
    change = abs(close_series - close_series.shift(10))
    volatility = abs(close_series.diff()).rolling(window=10).sum()
    er = change / volatility.replace(0, np.nan)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # KAMA direction: 1 if rising, -1 if falling
    kama_dir = np.where(kama > np.roll(kama, 1), 1, -1)
    kama_dir[0] = 0  # first value
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # ensure sufficient warmup for KAMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(kama[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        if i >= 20:
            vol_ma20 = np.mean(volume[i-20:i])
            vol_ok = volume[i] > 1.5 * vol_ma20
        else:
            vol_ok = False
        
        if position == 0:
            # Long: KAMA rising + close > KAMA + volume confirmation + above 1d EMA trend
            if kama_dir[i] == 1 and close[i] > kama[i] and vol_ok and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling + close < KAMA + volume confirmation + below 1d EMA trend
            elif kama_dir[i] == -1 and close[i] < kama[i] and vol_ok and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA falling or price crosses below KAMA
            if kama_dir[i] == -1 or close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA rising or price crosses above KAMA
            if kama_dir[i] == 1 or close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals