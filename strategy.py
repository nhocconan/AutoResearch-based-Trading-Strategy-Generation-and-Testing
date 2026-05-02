#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Uses 4h timeframe for signal generation with Donchian channel breakouts
# Daily trend filter (price > daily EMA50 for longs, < for shorts) ensures alignment with higher timeframe bias
# Volume confirmation (1.8x 20-period average) filters for institutional participation
# Discrete position sizing (0.25) balances profit potential with fee drag minimization
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Daily trend filter provides robustness in both bull and bear markets by avoiding counter-trend trades
# Williams Alligator is not used; instead relying on price structure and volume for clarity

name = "4h_Donchian20_1dEMA50_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_band = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    lower_band = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Daily trend filter: price > daily EMA50 for longs, < for shorts
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above upper band + volume spike + price > daily EMA50
            if close[i] > upper_band[i] and volume_spike[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower band + volume spike + price < daily EMA50
            elif close[i] < lower_band[i] and volume_spike[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below lower band or price < daily EMA50
            if close[i] < lower_band[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close above upper band or price > daily EMA50
            if close[i] > upper_band[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals