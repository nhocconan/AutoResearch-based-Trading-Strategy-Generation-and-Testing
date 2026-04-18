#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and volatility filter.
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue).
# Volume ensures institutional participation; volatility filter avoids chop.
# Target: 20-40 trades/year per symbol.
name = "4h_Donchian20_VolumeVolatilityFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for calculations (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily ATR (14-period) for volatility filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.roll(df_1d['close'], 1))
    tr3 = np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 20-period Donchian channels on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_1d_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR above its 50-period average (avoid low volatility chop)
        if i >= 50:
            atr_ma_50 = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
            atr_ma_50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50)
            vol_filter = not np.isnan(atr_ma_50_aligned[i]) and atr_1d_aligned[i] > atr_ma_50_aligned[i]
        else:
            vol_filter = False
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        trade_allowed = vol_filter and vol_confirm
        
        if position == 0:
            # Long: price breaks above 20-period high
            if trade_allowed and close[i] > highest_high[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-period low
            elif trade_allowed and close[i] < lowest_low[i-1]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 10-period low (faster exit)
            lowest_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
            if not np.isnan(lowest_low_10[i]) and close[i] < lowest_low_10[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above 10-period high
            highest_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
            if not np.isnan(highest_high_10[i]) and close[i] > highest_high_10[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals