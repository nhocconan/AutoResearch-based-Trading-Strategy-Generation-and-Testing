#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian(20) breakout with 1d ATR(14) volatility filter and volume confirmation
# Long when price breaks above 1d Donchian high AND 1d ATR(14) > 1.5 * ATR(50) (high volatility regime) AND volume > 1.3 * avg_volume(20) on 4h
# Short when price breaks below 1d Donchian low AND 1d ATR(14) > 1.5 * ATR(50) AND volume > 1.3 * avg_volume(20) on 4h
# Exit when price crosses back through the 1d Donchian midpoint (high+low)/2
# Uses discrete sizing 0.25 to balance return and risk
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Donchian channels from 1d provide structural breakout levels that work in both bull and bear markets
# ATR regime filter ensures we trade only during sufficient volatility, reducing false breakouts
# Volume confirmation validates breakout strength

name = "4h_1dDonchian20_ATRRegime_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Donchian and ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need at least 50 completed daily bars for ATR(50)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian(20) channels
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high = high_20
    donchian_low = low_20
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate 1d ATR(14) and ATR(50) for volatility regime filter
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First TR is undefined
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    volatility_regime = atr_14 > (1.5 * atr_50)  # High volatility regime
    
    # Align 1d indicators to 4h timeframe (wait for completed 1d bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    volatility_regime_aligned = align_htf_to_ltf(prices, df_1d, volatility_regime)
    
    # Calculate volume confirmation: volume > 1.3 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(volatility_regime_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Donchian high, high volatility regime, volume confirmation, in session
            if (close[i] > donchian_high_aligned[i] and 
                volatility_regime_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian low, high volatility regime, volume confirmation, in session
            elif (close[i] < donchian_low_aligned[i] and 
                  volatility_regime_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 1d Donchian midpoint
            if close[i] < donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above 1d Donchian midpoint
            if close[i] > donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals