#!/usr/bin/env python3
# Hypothesis: 1h price action with 4h trend filter and 1d volume regime
# Long when: price > 4h EMA50, 1d volume > 1.5x 20-day average, and RSI(14) > 55
# Short when: price < 4h EMA50, 1d volume > 1.5x 20-day average, and RSI(14) < 45
# Exit when RSI crosses back to 50 (neutral)
# Uses multi-timeframe alignment: 4h EMA for trend, 1d volume for regime filter, 1h RSI for entry timing
# Designed to capture momentum bursts during high-volume regimes while avoiding low-volatility chop
# Target: 60-150 total trades over 4 years (15-37/year) with size 0.20

name = "1h_EMA50_VolumeRegime_RSI"
timeframe = "1h"
leverage = 1.0

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
    
    # Calculate RSI(14) on 1h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 1d volume regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean()
    vol_ratio = df_1d['volume'] / (vol_ma_1d + 1e-10)
    vol_regime = vol_ratio > 1.5  # High volume regime
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_regime_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above 4h EMA50, high volume regime, RSI > 55
            if (close[i] > ema50_4h_aligned[i] and 
                vol_regime_aligned[i] and 
                rsi[i] > 55):
                signals[i] = 0.20
                position = 1
            # Enter short: price below 4h EMA50, high volume regime, RSI < 45
            elif (close[i] < ema50_4h_aligned[i] and 
                  vol_regime_aligned[i] and 
                  rsi[i] < 45):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral (50)
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI returns to neutral (50)
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals