#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian channel breakout with 1-day ATR volatility filter and volume confirmation
# Long when price breaks above 20-period high AND ATR(14) > 1.5x ATR(50) AND volume > 1.5x 20-period average
# Short when price breaks below 20-period low AND ATR(14) > 1.5x ATR(50) AND volume > 1.5x 20-period average
# Exit when price crosses back to the midline (average of 20-period high and low)
# Uses volatility expansion to catch breakouts in both bull and bear markets, with volume confirmation
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data ONCE before loop for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels (20-period high/low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate ATR on daily timeframe (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR on daily timeframe (50-period for comparison)
    atr50_1d = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Align ATR values to 4h timeframe
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    atr50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr50_1d)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr14_1d_aligned[i]) or np.isnan(atr50_1d_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        atr_condition = atr14_1d_aligned[i] > (1.5 * atr50_1d_aligned[i])
        
        if position == 0:
            # Long setup: break above Donchian high + volatility expansion + volume confirmation
            if (price > donchian_high[i] and atr_condition and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: break below Donchian low + volatility expansion + volume confirmation
            elif (price < donchian_low[i] and atr_condition and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to midline
            if price < donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to midline
            if price > donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_ATR_Volume"
timeframe = "4h"
leverage = 1.0