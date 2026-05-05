#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with 1w EMA34 filter and volume confirmation
# Long when: close > KAMA(10) AND close > 1w EMA34 AND volume > 1.5x 20-period MA
# Short when: close < KAMA(10) AND close < 1w EMA34 AND volume > 1.5x 20-period MA
# Exit when: price crosses KAMA in opposite direction
# Uses KAMA for adaptive trend, 1w EMA for major trend alignment, volume for conviction
# Timeframe: 1d, HTF: 1w. Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag.

name = "1d_KAMA_1wEMA34_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for EMA34
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # need sufficient data for EMA34
        return np.zeros(n)
    
    # Calculate 1w EMA34
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate KAMA(10) on 1d data
    if len(close) >= 10:
        # Efficiency Ratio (ER)
        change = np.abs(np.diff(close, n=10))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        # Smoothing constants
        fastest = 2 / (2 + 1)
        slowest = 2 / (30 + 1)
        sc = (er * (fastest - slowest) + slowest) ** 2
        # Initialize KAMA
        kama = np.full(n, np.nan)
        kama[9] = close[9]  # seed with first close
        for i in range(10, n):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    else:
        kama = np.full(n, np.nan)
    
    # Volume confirmation on 1d
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (1.5 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(10, n):
        # Skip if any value is NaN
        if (np.isnan(kama[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > KAMA AND above 1w EMA34 AND volume spike
            if (close[i] > kama[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < KAMA AND below 1w EMA34 AND volume spike
            elif (close[i] < kama[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below KAMA
            if close[i] <= kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above KAMA
            if close[i] >= kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals