#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA(50) trend filter and volume spike confirmation.
# Long when close breaks above Donchian upper band, 12h EMA(50) rising, volume > 2x 20-period average.
# Short when close breaks below Donchian lower band, 12h EMA(50) falling, volume > 2x average.
# Exit when price crosses back through Donchian midpoint or volume dries up.
# Designed for ~20-40 trades/year per symbol with strong trend capture and minimal whipsaw.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    dc_upper = high_20
    dc_lower = low_20
    dc_mid = (dc_upper + dc_lower) / 2.0
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 50-period EMA on 12h close for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume filter: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or 
            np.isnan(dc_mid[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: break above upper band, 12h EMA rising, volume spike
        if (close[i] > dc_upper[i] and 
            ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and 
            volume_filter[i]):
            signals[i] = 0.30
            position = 1
        # Short entry: break below lower band, 12h EMA falling, volume spike
        elif (close[i] < dc_lower[i] and 
              ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and 
              volume_filter[i]):
            signals[i] = -0.30
            position = -1
        # Exit conditions: price crosses midline or volume dries up
        elif position == 1:
            if close[i] < dc_mid[i] or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            if close[i] > dc_mid[i] or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_DonchianBreakout_12hEMA50_VolumeFilter"
timeframe = "4h"
leverage = 1.0