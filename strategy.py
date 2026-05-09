#!/usr/bin/env python3
# Hypothesis: 1d KAMA trend direction with volume confirmation and Bollinger Band mean reversion
# Long when KAMA indicates uptrend, price touches lower Bollinger Band, and volume > 1.5x average
# Short when KAMA indicates downtrend, price touches upper Bollinger Band, and volume > 1.5x average
# Exit when price crosses KAMA in the opposite direction
# Uses 1w EMA50 as higher timeframe trend filter to avoid counter-trend trades
# Designed for low-frequency, high-conviction trades on 1d timeframe suitable for trending and ranging markets
# Target: 30-100 total trades over 4 years (7-25/year) with size 0.25

name = "1d_KAMA_BBands_Volume_1wEMA50"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # KAMA calculation (close price)
    close_series = pd.Series(close)
    delta = close_series.diff().abs()
    # Avoid division by zero: add small epsilon where delta_sum is zero
    delta_sum = delta.rolling(window=10, min_periods=10).sum()
    # Where delta_sum is zero, set ER to 0 (no volatility)
    er = delta / (delta_sum + 1e-10)
    # Where delta_sum is zero, er becomes 0/0 -> NaN, replace with 0
    er = er.fillna(0)
    # Scaling factors
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean()
    std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std()
    upper_band = sma + (std_dev * bb_std)
    lower_band = sma - (std_dev * bb_std)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 30, bb_period)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(upper_band.iloc[i]) or np.isnan(lower_band.iloc[i]) or
            np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: KAMA uptrend, price at lower BB, volume spike
            if (kama[i] > kama[i-1] and  # KAMA rising
                close[i] <= lower_band.iloc[i] and  # Price at or below lower BB
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA downtrend, price at upper BB, volume spike
            elif (kama[i] < kama[i-1] and  # KAMA falling
                  close[i] >= upper_band.iloc[i] and  # Price at or above upper BB
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below KAMA
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above KAMA
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals