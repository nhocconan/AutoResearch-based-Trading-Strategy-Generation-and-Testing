#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d volatility filter and volume confirmation
# Long when price breaks above 20-period Donchian upper band AND 1d ATR ratio > 1.2 (high volatility regime) AND volume > 1.5x average
# Short when price breaks below 20-period Donchian lower band AND 1d ATR ratio > 1.2 AND volume > 1.5x average
# Exit when price crosses the 10-period SMA (opposite direction) OR ATR ratio drops below 0.8 (low volatility)
# Donchian captures breakouts; 1d ATR regime filter ensures trades only in high volatility environments; volume confirms conviction
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for volatility filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels (20-period)
    dc_upper = pd.Series(high).rolling(window=20, min_periods=20).max()
    dc_lower = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # Calculate 10-period SMA for exit
    sma_10 = pd.Series(close).rolling(window=10, min_periods=10).mean()
    
    # Calculate ATR on 1d for volatility regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    
    # Calculate ATR ratio (current ATR / 50-period average ATR) to detect volatility regime
    atr_ma_50 = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean()
    atr_ratio = atr_1d / atr_ma_50
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(dc_upper[i]) or 
            np.isnan(dc_lower[i]) or 
            np.isnan(sma_10[i]) or 
            np.isnan(atr_ratio[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        # Get ATR ratio values aligned to 12h timeframe
        atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio.values)
        atr_ratio_val = atr_ratio_aligned[i]
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: Price breaks above Donchian upper AND high volatility regime (ATR ratio > 1.2) AND volume confirmation
            if (price > dc_upper[i] and atr_ratio_val > 1.2 and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: Price breaks below Donchian lower AND high volatility regime AND volume confirmation
            elif (price < dc_lower[i] and atr_ratio_val > 1.2 and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price crosses below 10-period SMA OR volatility drops (ATR ratio < 0.8)
            if (price < sma_10[i] or atr_ratio_val < 0.8):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Price crosses above 10-period SMA OR volatility drops
            if (price > sma_10[i] or atr_ratio_val < 0.8):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Donchian_1dATR_Volume"
timeframe = "12h"
leverage = 1.0