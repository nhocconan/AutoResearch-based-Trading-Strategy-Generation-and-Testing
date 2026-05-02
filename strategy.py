#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h HMA21 trend filter and volume confirmation
# Donchian breakouts capture strong momentum; 12h HMA21 ensures smooth intermediate trend alignment
# Volume spike (2.5x 20-period average) confirms institutional participation and reduces false breakouts
# Discrete position sizing (0.25) minimizes fee churn
# Targets 19-50 trades/year (75-200 total over 4 years) for 4h timeframe
# Works in bull markets via breakout continuation and in bear markets via filtered short breakdowns

name = "4h_Donchian20_12hHMA21_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for HMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate 12h HMA(21) for trend filter
    def calculate_hma(arr, period):
        half = arr.copy()
        half[:] = np.nan
        half[period//2:] = arr[:-period//2] if period//2 > 0 else arr
        wma2 = pd.Series(half).ewm(span=period//2, adjust=False, min_periods=period//2).mean().values
        wma1 = pd.Series(arr).ewm(span=period//2, adjust=False, min_periods=period//2).mean().values
        raw_hma = 2 * wma1 - wma2
        hma = pd.Series(raw_hma).ewm(span=int(np.sqrt(period)), adjust=False, min_periods=int(np.sqrt(period))).mean().values
        return hma
    
    hma_12h = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate Donchian(20) channels from 4h data
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate volume spike (2.5x 20-period average) - higher threshold to reduce trades
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian calculation and volume MA)
    start_idx = 20  # buffer for 20-period calculations
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(hma_12h_aligned[i]) or np.isnan(high_ma[i]) or 
            np.isnan(low_ma[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper + 12h close > HMA21 + volume spike
            if (close[i] > high_ma[i] and 
                close[i] > hma_12h_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower + 12h close < HMA21 + volume spike
            elif (close[i] < low_ma[i] and 
                  close[i] < hma_12h_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price drops below Donchian lower (reversal to support) or 12h trend breaks
            if close[i] < low_ma[i] or close[i] < hma_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above Donchian upper (reversal to resistance) or 12h trend breaks
            if close[i] > high_ma[i] or close[i] > hma_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals