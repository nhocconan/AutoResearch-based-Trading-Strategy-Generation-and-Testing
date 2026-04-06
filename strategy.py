#!/usr/bin/env python3
"""
6h/1d Camarilla Pivot Fade + Volume Confirmation
Hypothesis: Camarilla pivot levels act as mean reversion zones in ranging markets and breakout confirmers in trending markets. At R3/S3 we fade (counter-trend), at R4/S4 we continue (trend). Volume confirms institutional interest. Designed for 6h timeframe with 1d pivot context to work in both bull (trend continuation) and bear (mean reversion at extremes) markets. Targets 50-150 trades over 4 years via strict pivot-level filtering.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR for stoploss
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        atr[0] = np.nan
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # Using previous day's data to avoid look-ahead
    phigh = df_1d['high'].shift(1).values  # Previous day high
    plow = df_1d['low'].shift(1).values    # Previous day low
    pclose = df_1d['close'].shift(1).values # Previous day close
    
    # Calculate pivot levels
    camarilla_r4 = pclose + ((phigh - plow) * 1.1 / 2)
    camarilla_r3 = pclose + ((phigh - plow) * 1.1 / 4)
    camarilla_s3 = pclose - ((phigh - plow) * 1.1 / 4)
    camarilla_s4 = pclose - ((phigh - plow) * 1.1 / 2)
    
    # Align to 6h timeframe (shifted by 1 day already in calculation)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 2  # Need at least 2 days for previous day data
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or \
           np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter (20-period average)
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            volume_filter = volume[i] > vol_ma * 1.5
        else:
            volume_filter = False
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price reaches S3 (mean reversion target) or stoploss
            if (close[i] <= s3_aligned[i] or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches R3 (mean reversion target) or stoploss
            if (close[i] >= r3_aligned[i] or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries at extreme levels with volume confirmation
            # At R4/S4: breakout continuation (trend following)
            # At R3/S3: mean reversion fade
            
            # Breakout continuation at R4/S4
            if volume_filter:
                if close[i] > r4_aligned[i]:  # Break above R4 -> long
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif close[i] < s4_aligned[i]:  # Break below S4 -> short
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            # Mean reversion fade at R3/S3 (only if not at extreme breakout levels)
            elif volume_filter:
                if close[i] < r3_aligned[i] and close[i] > r4_aligned[i]:  # Between R3 and R4 -> fade short
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                elif close[i] > s3_aligned[i] and close[i] < s4_aligned[i]:  # Between S3 and S4 -> fade long
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals