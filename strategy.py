#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR filter and volume confirmation
# Long when: price breaks above 20-period high, volume > 1.5x 20-period average, and 1d ATR(14) < 0.05 * price (low volatility breakout)
# Short when: price breaks below 20-period low, volume > 1.5x 20-period average, and 1d ATR(14) < 0.05 * price
# Exit when price returns to 20-period midpoint or opposite breakout
# Uses Donchian channels for structure, effective in trending markets with volatility filter to avoid false breakouts in chop.
# Timeframe: 4h, HTF: 1d. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Donchian20_Breakout_1dATR_Filter_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Calculate Donchian channels on 4h (20-period)
    if len(high) >= 20:
        # Rolling max/min for Donchian channels
        high_rolling_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
        low_rolling_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (high_rolling_max + low_rolling_min) / 2.0
    else:
        high_rolling_max = np.full(n, np.nan)
        low_rolling_min = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Calculate volume confirmation on 4h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) for volatility filter
    if len(high_1d) >= 15:
        # True Range calculation
        tr1 = high_1d[1:] - low_1d[1:]
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        # Prepend NaN for first element
        tr = np.concatenate([[np.nan], tr])
        # ATR as EMA of TR (Wilder's smoothing)
        atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
        # Normalize ATR by price to get percentage
        atr_pct = atr_14 / close_1d
        # Filter: low volatility (ATR < 5% of price)
        vol_filter_1d = atr_pct < 0.05
    else:
        atr_14 = np.full(len(close_1d), np.nan)
        vol_filter_1d = np.zeros(len(close_1d), dtype=bool)
    
    # Align 1d volatility filter to 4h timeframe
    vol_filter_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_filter_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any value is NaN
        if (np.isnan(high_rolling_max[i]) or 
            np.isnan(low_rolling_min[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(vol_filter_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high, volume filter, and low volatility regime
            if (close[i] > high_rolling_max[i] and 
                open_price[i] <= high_rolling_max[i] and  # Ensure breakout happens on this bar
                volume_filter[i] and 
                vol_filter_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low, volume filter, and low volatility regime
            elif (close[i] < low_rolling_min[i] and 
                  open_price[i] >= low_rolling_min[i] and  # Ensure breakdown happens on this bar
                  volume_filter[i] and 
                  vol_filter_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to midpoint or breaks below low (reversal)
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midpoint or breaks above high (reversal)
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals