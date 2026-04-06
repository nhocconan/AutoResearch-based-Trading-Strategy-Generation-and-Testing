#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla pivot levels from daily data with volume confirmation.
# Fade at R3/S3 levels (mean reversion), breakout continuation at R4/S4 levels (trend follow).
# Uses weekly trend filter (price > weekly EMA200 for longs, < for shorts) to avoid counter-trend trades.
# Volume confirmation: current volume > 1.5x 20-period average.
# Stoploss at 2 * ATR(14). Position size: 0.25.
# Target: 80-180 trades over 4 years (20-45/year) - balances mean reversion and trend following.

name = "6h_camarilla_1d_weekly_ema200_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Typical Price = (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Pivot = Typical Price
    pivot = typical_price.values
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    r1 = df_1d['close'] + (df_1d['high'] - df_1d['low']) * 1.1 / 12
    s1 = df_1d['close'] - (df_1d['high'] - df_1d['low']) * 1.1 / 12
    # R2 = C + (H-L)*1.1/6, S2 = C - (H-L)*1.1/6
    r2 = df_1d['close'] + (df_1d['high'] - df_1d['low']) * 1.1 / 6
    s2 = df_1d['close'] - (df_1d['high'] - df_1d['low']) * 1.1 / 6
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    r3 = df_1d['close'] + (df_1d['high'] - df_1d['low']) * 1.1 / 4
    s3 = df_1d['close'] - (df_1d['high'] - df_1d['low']) * 1.1 / 4
    # R4 = C + (H-L)*1.1/2, S4 = C - (H-L)*1.1/2
    r4 = df_1d['close'] + (df_1d['high'] - df_1d['low']) * 1.1 / 2
    s4 = df_1d['close'] - (df_1d['high'] - df_1d['low']) * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (use previous day's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
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
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below S3 (mean reversion fail) or goes above R4 (take profit)
            elif close[i] < s3_aligned[i] or close[i] > r4_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above R3 (mean reversion fail) or goes below S4 (take profit)
            elif close[i] > r3_aligned[i] or close[i] < s4_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long fade at S3: price < S3 and > S4, weekly bullish
            if (close[i] < s3_aligned[i] and close[i] > s4_aligned[i] and
                close[i] > ema_200_1w_aligned[i] and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short fade at R3: price > R3 and < R4, weekly bearish
            elif (close[i] > r3_aligned[i] and close[i] < r4_aligned[i] and
                  close[i] < ema_200_1w_aligned[i] and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            # Long breakout at R4: price > R4, weekly bullish
            elif (close[i] > r4_aligned[i] and
                  close[i] > ema_200_1w_aligned[i] and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short breakdown at S4: price < S4, weekly bearish
            elif (close[i] < s4_aligned[i] and
                  close[i] < ema_200_1w_aligned[i] and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals