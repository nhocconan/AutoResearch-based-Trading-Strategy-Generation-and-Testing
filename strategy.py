#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour KAMA with 1-day RSI and volume confirmation
# Long when KAMA > KAMA(1) and RSI(14) > 50 and volume > 1.2x 20-period average
# Short when KAMA < KAMA(1) and RSI(14) < 50 and volume > 1.2x 20-period average
# Exit when KAMA crosses opposite direction or volume drops below threshold
# Uses adaptive trend filter (KAMA) with momentum filter (RSI) and volume confirmation
# Designed to work in both bull and bear markets by requiring volume confirmation and momentum alignment

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for RSI filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate KAMA on 4h (ER=10, fast=2, slow=30)
    close_series = pd.Series(close)
    change = abs(close_series.diff(10))
    volatility = abs(close_series.diff(1)).rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    sc = sc.fillna(0)
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI on 1d (14-period)
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_values = rsi_14.fillna(50).values
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d RSI to 4h timeframe
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_values)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi_14_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        # Get previous KAMA for crossover detection
        kama_prev = kama[i-1] if i > 0 else kama[0]
        
        price = close[i]
        rsi = rsi_14_aligned[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.2
        
        if position == 0:
            # Long setup: KAMA bullish crossover + RSI > 50 + volume confirmation
            if (kama[i] > kama_prev and rsi > 50 and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: KAMA bearish crossover + RSI < 50 + volume confirmation
            elif (kama[i] < kama_prev and rsi < 50 and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: KAMA bearish crossover OR volume drops below threshold
            if (kama[i] < kama_prev) or (vol < vol_threshold):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: KAMA bullish crossover OR volume drops below threshold
            if (kama[i] > kama_prev) or (vol < vol_threshold):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_KAMA_1dRSI_Volume"
timeframe = "4h"
leverage = 1.0