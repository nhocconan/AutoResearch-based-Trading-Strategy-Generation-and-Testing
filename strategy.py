#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Camarilla pivot levels with volume confirmation
# Camarilla levels provide structured support/resistance based on prior day's range
# Fade at R3/S3 levels (mean reversion in ranging markets)
# Breakout continuation at R4/S4 levels (trend following in strong moves)
# Volume filter ensures breakouts/ reversals have participation
# Works in bull/bear markets: mean reversion in ranges, trend following on breakouts
# Target: 12-37 trades/year via tight entry conditions at specific pivot levels

name = "6h_1d_camarilla_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels
    # Based on prior day's OHLC
    pivot = (high_1d[:-1] + low_1d[:-1] + close_1d[:-1]) / 3.0
    range_1d = high_1d[:-1] - low_1d[:-1]
    
    # Camarilla levels
    r4 = pivot + (range_1d * 1.1 / 2)
    r3 = pivot + (range_1d * 1.1 / 4)
    s3 = pivot - (range_1d * 1.1 / 4)
    s4 = pivot - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (using prior day's levels)
    r4_6h = align_htf_to_ltf(prices, df_1d[:-1], r4)  # shift by 1 to avoid look-ahead
    r3_6h = align_htf_to_ltf(prices, df_1d[:-1], r3)
    s3_6h = align_htf_to_ltf(prices, df_1d[:-1], s3)
    s4_6h = align_htf_to_ltf(prices, df_1d[:-1], s4)
    
    # Calculate volume ratio (current vs 20-period average)
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: need above average volume
        vol_confirmed = vol_ratio[i] > 1.2
        
        if position == 1:  # Long position
            # Exit long if price reaches R4 (take profit) or breaks below S3 (stop)
            if close[i] >= r4_6h[i] or close[i] < s3_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit short if price reaches S4 (take profit) or breaks above R3 (stop)
            if close[i] <= s4_6h[i] or close[i] > r3_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Fade at R3/S3 with volume (mean reversion)
            if close[i] > r3_6h[i] and vol_confirmed:
                position = -1
                signals[i] = -0.25
            elif close[i] < s3_6h[i] and vol_confirmed:
                position = 1
                signals[i] = 0.25
            # Breakout continuation at R4/S4 with volume (trend following)
            elif close[i] > r4_6h[i] and vol_confirmed:
                position = 1
                signals[i] = 0.25
            elif close[i] < s4_6h[i] and vol_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals