#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h 14-day Donchian breakout with 1-day high/low filter and volume confirmation
# In bull markets: buy breakouts above 14-day high when above 1-day low (uptrend filter)
# In bear markets: sell breakdowns below 14-day low when below 1-day high (downtrend filter)
# Volume filter ensures breakouts have participation
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE for 1-day high/low filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day high/low (trend filter)
    high_1d_val = high_1d[-1]  # current day's high
    low_1d_val = low_1d[-1]    # current day's low
    # Since we need the previous day's high/low for filtering (completed day)
    # We'll shift by 1 to get the previous completed day's values
    high_1d_prev = np.concatenate([[np.nan], high_1d[:-1]])
    low_1d_prev = np.concatenate([[np.nan], low_1d[:-1]])
    
    # Load daily data ONCE for 14-day Donchian channels
    high_14d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    low_14d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Align daily Donchian channels to 12h timeframe
    high_14d_aligned = align_htf_to_ltf(prices, df_1d, high_14d)
    low_14d_aligned = align_htf_to_ltf(prices, df_1d, low_14d)
    
    # Calculate 12h ATR for volatility filter and stop sizing
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Precompute hour of day for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Volume filter: 12h volume > 20-period average
    volume = prices['volume'].values
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(high_14d_aligned[i]) or np.isnan(low_14d_aligned[i]) or \
           np.isnan(high_1d_prev[i]) or np.isnan(low_1d_prev[i]) or np.isnan(atr_12h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter
        vol_filter = volume[i] > volume_ma_20[i]
        
        # Price levels
        resistance = high_14d_aligned[i]
        support = low_14d_aligned[i]
        daily_high = high_1d_prev[i]   # previous day's high
        daily_low = low_1d_prev[i]     # previous day's low
        price = close[i]
        
        if position == 0:
            # Long: price breaks above 14-day resistance, above previous day's low (uptrend), with volume
            if price > resistance and price > daily_low and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below 14-day support, below previous day's high (downtrend), with volume
            elif price < support and price < daily_high and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: stop loss (2x ATR below entry) or price breaks below 14-day support
            if price <= entry_price - 2.0 * atr_12h[i] or price < support:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stop loss (2x ATR above entry) or price breaks above 14-day resistance
            if price >= entry_price + 2.0 * atr_12h[i] or price > resistance:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_14D_Donchian_1D_Trend_VolumeFilter"
timeframe = "12h"
leverage = 1.0