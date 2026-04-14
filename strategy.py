#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day KAMA with RSI filter and volume confirmation.
# Long when price crosses above KAMA, RSI > 50, and volume > 1.3x average.
# Short when price crosses below KAMA, RSI < 50, and volume > 1.3x average.
# Exit when price crosses back across KAMA or volume drops below average.
# Uses KAMA for adaptive trend following, RSI for momentum confirmation,
# and volume for institutional participation. Designed to work in both bull and bear
# markets by only trading when momentum aligns with trend (RSI >50 for long, <50 for short).
# Target: 15-30 trades/year per symbol (60-120 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for KAMA and RSI periods
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate KAMA (10, 2, 30)
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.zeros_like(close_1d, dtype=np.float64)
    er[1:] = change[1:] / np.where(volatility[1:] == 0, 1, volatility[1:])
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close_1d, dtype=np.float64)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI (14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss == 0, 0, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # Align indicators to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume confirmation: 1.3x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20)  # Need KAMA/RSI and volume MA periods
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Look for KAMA crossovers with RSI filter
            # Long: price crosses above KAMA AND RSI > 50 AND volume confirmation
            if (close[i] > kama_aligned[i] and 
                close[i-1] <= kama_aligned[i-1] and 
                rsi_aligned[i] > 50 and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price crosses below KAMA AND RSI < 50 AND volume confirmation
            elif (close[i] < kama_aligned[i] and 
                  close[i-1] >= kama_aligned[i-1] and 
                  rsi_aligned[i] < 50 and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back below KAMA or volume drops
            if (close[i] < kama_aligned[i] or 
                not volume_confirmed):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses back above KAMA or volume drops
            if (close[i] > kama_aligned[i] or 
                not volume_confirmed):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_KAMA_RSI_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0