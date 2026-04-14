#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Channel Breakout with 1d ATR Filter and Volume Spike
# Uses Donchian Channel (20-period) for breakout entries in direction of higher timeframe trend
# 1d ATR-based volatility filter ensures trades occur during elevated volatility regimes
# Volume confirmation (>1.5x average) ensures institutional participation
# Designed to work in both bull and bear markets by trading breakouts in direction of higher timeframe trend
# Target: 25-50 trades/year (100-200 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d ATR data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR(14) on 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Donchian Channel (20-period) on 4h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_donchian = high_series.rolling(window=20, min_periods=20).max().values
    lower_donchian = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for 1d ATR and Donchian calculation
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(upper_donchian[i]) or 
            np.isnan(lower_donchian[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Volatility filter: only trade when ATR is above its 50-period average
        # Calculate 50-period average of ATR for volatility regime filter
        if i >= 50 + 50:  # Need enough history for ATR average
            atr_avg = np.nanmean(atr_14_1d_aligned[i-50:i])
            vol_filter = atr_14_1d_aligned[i] > 0.8 * atr_avg  # Trade when volatility is above 80% of recent average
        else:
            vol_filter = True  # Not enough data yet, allow trade
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume filter and volatility filter
            if price > upper_donchian[i] and vol > 1.5 * avg_vol[i] and vol_filter:
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower Donchian with volume filter and volatility filter
            elif price < lower_donchian[i] and vol > 1.5 * avg_vol[i] and vol_filter:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below lower Donchian (contrarian exit) or trailing stop
            if price < lower_donchian[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above upper Donchian (contrarian exit) or trailing stop
            if price > upper_donchian[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_Breakout_1dATR_Volume"
timeframe = "4h"
leverage = 1.0