# 4h_1d_camarilla_breakout_volume_v1
# Strategy: 4h Camarilla Pivot Level breakout with volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels identify key support/resistance zones from the prior daily session.
# Price breaking above/below these levels with volume confirmation indicates institutional interest.
# Works in bull/bear markets as pivots adapt to recent price action. Low trade frequency (~20-40/year)
# to minimize fee drag. Uses 1d OHLC to calculate Camarilla levels for the current 4h candle.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day using prior day's OHLC
    # Camarilla formulas:
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.0 * (high - low)
    # H2 = close + 0.5 * (high - low)
    # H1 = close + 0.25 * (high - low)
    # L1 = close - 0.25 * (high - low)
    # L2 = close - 0.5 * (high - low)
    # L3 = close - 1.0 * (high - low)
    # L4 = close - 1.5 * (high - low)
    
    # We'll use H3/L3 as primary breakout levels (1.0 * range)
    # And H4/L4 as stronger confirmation (1.5 * range)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate levels for each day
    range_1d = high_1d - low_1d
    h3 = close_1d + 1.0 * range_1d
    l3 = close_1d - 1.0 * range_1d
    h4 = close_1d + 1.5 * range_1d
    l4 = close_1d - 1.5 * range_1d
    
    # Align to 4h timeframe (use prior day's levels for current day's trading)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry logic: Camarilla breakout + volume confirmation
        # Strong breakout: beyond H4/L4
        # Regular breakout: beyond H3/L3 with volume
        
        if close[i] > h4_aligned[i] and volume[i] > 0 and position != 1:
            # Strong bullish breakout
            position = 1
            signals[i] = 0.30
        elif close[i] < l4_aligned[i] and volume[i] > 0 and position != -1:
            # Strong bearish breakout
            position = -1
            signals[i] = -0.30
        elif close[i] > h3_aligned[i] and vol_confirm[i] and position != 1:
            # Regular bullish breakout with volume
            position = 1
            signals[i] = 0.25
        elif close[i] < l3_aligned[i] and vol_confirm[i] and position != -1:
            # Regular bearish breakout with volume
            position = -1
            signals[i] = -0.25
        # Exit: price returns to the opposite side of H3/L3
        elif position == 1 and close[i] < l3_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > h3_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.30 if position == 1 else (-0.30 if position == -1 else 0.0)
    
    return signals