#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot levels from 1-day data with volume confirmation
# Fade at R3/S3 levels (mean reversion) when price reaches these levels with volume confirmation
# Breakout continuation at R4/S4 levels (trend following) when price breaks these levels with volume confirmation
# Uses 1-day Camarilla levels calculated from previous day's high, low, close
# Position size: 0.25 (25% of capital)
# Target: 80-180 total trades over 4 years (20-45/year)

name = "6h_camarilla_1d_fade_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R4 = Close + (High - Low) * 1.1/2
    # R3 = Close + (High - Low) * 1.1/4
    # S3 = Close - (High - Low) * 1.1/4
    # S4 = Close - (High - Low) * 1.1/2
    camarilla_r4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    camarilla_s4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: 20-period moving average of volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price reaches S3 (take profit) or breaks below S4 (stop and reverse)
            elif close[i] <= s3_aligned[i]:
                signals[i] = 0.0  # take profit at S3
                position = 0
                entry_price = 0.0
            elif close[i] < s4_aligned[i]:
                signals[i] = -0.25  # reverse to short at S4 break
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price reaches R3 (take profit) or breaks above R4 (stop and reverse)
            elif close[i] >= r3_aligned[i]:
                signals[i] = 0.0  # take profit at R3
                position = 0
                entry_price = 0.0
            elif close[i] > r4_aligned[i]:
                signals[i] = 0.25  # reverse to long at R4 break
                position = 1
                entry_price = close[i]
            else:
                signals[i] = -0.25
        else:
            # Look for fade entries at R3/S3 with volume confirmation
            # Fade long at S3: price touches S3 and bounces up with volume
            if (close[i] <= s3_aligned[i] * 1.001 and  # allow small buffer
                close[i] > open[i] and  # bullish candle
                volume[i] > volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Fade short at R3: price touches R3 and bounces down with volume
            elif (close[i] >= r3_aligned[i] * 0.999 and  # allow small buffer
                  close[i] < open[i] and  # bearish candle
                  volume[i] > volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            # Look for breakout entries at R4/S4 with volume confirmation
            # Breakout long: price breaks above R4 with volume
            elif (close[i] > r4_aligned[i] and
                  open[i] <= r4_aligned[i] and  # was below or at R4
                  volume[i] > volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Breakout short: price breaks below S4 with volume
            elif (close[i] < s4_aligned[i] and
                  open[i] >= s4_aligned[i] and  # was above or at S4
                  volume[i] > volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals