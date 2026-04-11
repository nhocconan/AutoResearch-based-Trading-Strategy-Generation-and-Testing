#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    range_1d = high_1d - low_1d
    # Camarilla levels based on previous day's range
    H5 = close_1d + range_1d * 1.1/2
    H4 = close_1d + range_1d * 1.1/4
    H3 = close_1d + range_1d * 1.1/6
    L3 = close_1d - range_1d * 1.1/6
    L4 = close_1d - range_1d * 1.1/4
    L5 = close_1d - range_1d * 1.1/2
    
    # Shift by 1 to use only completed daily bars
    H5 = np.roll(H5, 1)
    H4 = np.roll(H4, 1)
    H3 = np.roll(H3, 1)
    L3 = np.roll(L3, 1)
    L4 = np.roll(L4, 1)
    L5 = np.roll(L5, 1)
    H5[0] = np.nan
    H4[0] = np.nan
    H3[0] = np.nan
    L3[0] = np.nan
    L4[0] = np.nan
    L5[0] = np.nan
    
    # Align daily levels to 4h timeframe
    H5_4h = align_htf_to_ltf(prices, df_1d, H5)
    H4_4h = align_htf_to_ltf(prices, df_1d, H4)
    H3_4h = align_htf_to_ltf(prices, df_1d, H3)
    L3_4h = align_htf_to_ltf(prices, df_1d, L3)
    L4_4h = align_htf_to_ltf(prices, df_1d, L4)
    L5_4h = align_htf_to_ltf(prices, df_1d, L5)
    
    # Volume filter: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(H5_4h[i]) or np.isnan(H4_4h[i]) or np.isnan(H3_4h[i]) or
            np.isnan(L3_4h[i]) or np.isnan(L4_4h[i]) or np.isnan(L5_4h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.8 * vol_ma
        
        # Long when price breaks above H4 with volume
        long_signal = volume_confirmed and (price_high > H4_4h[i])
        # Short when price breaks below L4 with volume
        short_signal = volume_confirmed and (price_low < L4_4h[i])
        
        # Exit when price returns to H3/L3 levels
        exit_long = price_low < H3_4h[i]
        exit_short = price_high > L3_4h[i]
        
        # Track position state
        if i == 20:
            position = 0
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Camarilla breakout strategy on 4h using daily pivot levels.
# Enters long when price breaks above H4 (1.1/4) level with volume confirmation (>1.8x average).
# Enters short when price breaks below L4 level with volume confirmation.
# Exits when price returns to H3/L3 levels, capturing mean reversion within the day's range.
# Uses volume filter to avoid false breakouts. Designed for 75-200 trades over 4 years
# (19-50/year) to minimize fee drag. Works in both bull and bear markets by capturing
# intraday momentum within the daily range structure. Camarilla levels provide
# mathematically derived support/resistance based on previous day's range.