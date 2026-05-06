#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian breakout with volume confirmation and ATR filter
# - Long when price breaks above 1d Donchian(20) high with volume spike and ATR > 0.5*ATR(50)
# - Short when price breaks below 1d Donchian(20) low with volume spike and ATR > 0.5*ATR(50)
# - Exit when price crosses the 1d Donchian midline (average of high/low over 20)
# - Volume filter requires current volume > 1.5x 20-period average
# - Designed to capture strong trending moves with volatility filter to avoid chop
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing
# - Works in both bull and breakouts: volatility filter avoids false signals in low-vol regimes

name = "12h_DonchianBreakout_1dVol_ATR"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian and ATR calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian(20) channels
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate ATR(50) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Align 1d indicators to 12h timeframe
    donchian_high_12h = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_12h = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_12h = align_htf_to_ltf(prices, df_1d, donchian_mid)
    atr_50_12h = align_htf_to_ltf(prices, df_1d, atr_50)
    
    # Volume filters (12h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high_12h[i]) or np.isnan(donchian_low_12h[i]) or 
            np.isnan(donchian_mid_12h[i]) or np.isnan(atr_50_12h[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with volume spike and sufficient volatility
            if high[i] > donchian_high_12h[i] and volume_spike[i] and atr_50_12h[i] > 0:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume spike and sufficient volatility
            elif low[i] < donchian_low_12h[i] and volume_spike[i] and atr_50_12h[i] > 0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian midline
            if close[i] < donchian_mid_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian midline
            if close[i] > donchian_mid_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals