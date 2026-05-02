#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ATR filter and volume confirmation
# Uses Donchian channel (20-period high/low) from 1d HTF for institutional support/resistance
# 1d ATR(14) filter ensures sufficient volatility to avoid whipsaws in low-volatility regimes
# Volume spike (1.8x 20-bar MA) confirms institutional participation
# Designed for 50-150 total trades over 4 years (12-37/year) on 12h timeframe
# Works in bull markets (breakout continuation) and bear markets (mean reversion at extremes)
# BTC and ETH focused with SOL as secondary validation

name = "12h_Donchian20_1dATR14_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First bar: no previous close
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # First bar: |high - close|
    tr3[0] = np.abs(low_1d[0] - close_1d[0])   # First bar: |low - close|
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Volume confirmation: 1.8x 20-period average (20*12h = 10 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Calculate 1d Donchian(20) levels
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 12h timeframe (wait for 1d close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for ATR14 and Donchian)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: require ATR > 0.5% of price to avoid choppy markets
        volatility_filter = atr_14_1d_aligned[i] > (0.005 * close[i])
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian high AND volume spike AND sufficient volatility
            if (close[i] > donchian_high_aligned[i] and 
                volume_spike[i] and 
                volatility_filter):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below Donchian low AND volume spike AND sufficient volatility
            elif (close[i] < donchian_low_aligned[i] and 
                  volume_spike[i] and 
                  volatility_filter):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Donchian low (mean reversion) OR volatility drops
            if close[i] < donchian_low_aligned[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian high (mean reversion) OR volatility drops
            if close[i] > donchian_high_aligned[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals