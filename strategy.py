#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Calculate Camarilla levels from previous day
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Camarilla levels (based on previous day's range)
    # H4 = Close + 1.5 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    # H3 = Close + 1.1 * (High - Low)
    # L3 = Close - 1.1 * (High - Low)
    # H2 = Close + 0.55 * (High - Low)
    # L2 = Close - 0.55 * (High - Low)
    range_1d = high_1d - low_1d
    h4 = close_1d + 1.5 * range_1d
    l4 = close_1d - 1.5 * range_1d
    h3 = close_1d + 1.1 * range_1d
    l3 = close_1d - 1.1 * range_1d
    h2 = close_1d + 0.55 * range_1d
    l2 = close_1d - 0.55 * range_1d
    
    # Shift by 1 to use only completed day's levels (previous day)
    h4 = np.roll(h4, 1)
    l4 = np.roll(l4, 1)
    h3 = np.roll(h3, 1)
    l3 = np.roll(l3, 1)
    h2 = np.roll(h2, 1)
    l2 = np.roll(l2, 1)
    h4[0] = np.nan
    l4[0] = np.nan
    h3[0] = np.nan
    l3[0] = np.nan
    h2[0] = np.nan
    l2[0] = np.nan
    
    # Align to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h2_aligned = align_htf_to_ltf(prices, df_1d, h2)
    l2_aligned = align_htf_to_ltf(prices, df_1d, l2)
    
    # Volume filter: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Trend filter: 4h EMA50
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h2_aligned[i]) or np.isnan(l2_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(ema_50[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.8 * vol_ma
        
        # Long breakout above H3 with volume
        long_signal = volume_confirmed and (price_close > h3_aligned[i]) and (price_close > ema_50[i])
        
        # Short breakdown below L3 with volume
        short_signal = volume_confirmed and (price_close < l3_aligned[i]) and (price_close < ema_50[i])
        
        # Exit when price returns to H2/L2 or reverses trend
        exit_long = position == 1 and (price_close < h2_aligned[i] or price_close < ema_50[i])
        exit_short = position == -1 and (price_close > l2_aligned[i] or price_close > ema_50[i])
        
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

# Hypothesis: Camarilla breakout strategy on 4h with volume confirmation and EMA50 trend filter.
# Uses Camarilla levels (H3, L3, H2, L2) calculated from previous day's range.
# Enters long when price breaks above H3 with volume >1.8x 20-period average and price above EMA50.
# Enters short when price breaks below L3 with volume confirmation and price below EMA50.
# Exits when price returns to H2/L2 or crosses EMA50 in opposite direction.
# This strategy aims for 20-50 trades per year on 4h timeframe, targeting 80-200 total trades over 4 years.
# The combination of price level breakout, volume confirmation, and trend filter reduces false signals
# and works in both bull and bear markets by following the intraday trend aligned with daily levels.