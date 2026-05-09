#!/usr/bin/env python3
# Hypothesis: 4h 4-period RSI mean reversion with 1d ATR-based volatility filter and volume confirmation
# Long when RSI < 30 and volatility above average, short when RSI > 70 and volatility above average
# Uses volatility filter to avoid ranging markets and focus on momentum exhaustion points
# Designed to work in both bull and bear markets by capturing overextended moves
# Target: 100-180 total trades over 4 years (25-45/year) with size 0.25

name = "4h_RSI_MeanReversion_VolatilityFilter_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4-period RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/4, adjust=False, min_periods=4).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/4, adjust=False, min_periods=4).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1d ATR for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First TR
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean()
    
    # ATR volatility filter: current ATR > 20-period average
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean()
    vol_filter = atr > atr_ma.values
    
    # Align RSI and volatility filter to 4h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi.values)
    vol_filter_aligned = align_htf_to_ltf(prices, df_1d, vol_filter.values)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi_aligned[i]) or np.isnan(vol_filter_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: RSI oversold (<30) with volatility expansion and volume confirmation
            if (rsi_aligned[i] < 30 and 
                vol_filter_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: RSI overbought (>70) with volatility expansion and volume confirmation
            elif (rsi_aligned[i] > 70 and 
                  vol_filter_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral (50) or overbought (>70)
            if rsi_aligned[i] >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral (50) or oversold (<30)
            if rsi_aligned[i] <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals