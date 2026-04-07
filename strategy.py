#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla pivot levels with 1-day EMA200 trend filter and volume confirmation
# Long when price breaks above Camarilla R4 level, 1d close > 1d EMA200 (uptrend), and volume > 1.5x 6s average volume
# Short when price breaks below Camarilla S4 level, 1d close < 1d EMA200 (downtrend), and volume > 1.5x 6s average volume
# Exit when trend reverses (1d close crosses EMA200) or price retests Camarilla R3/S3 levels
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses Camarilla pivots from 1-day timeframe for key support/resistance levels
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_camarilla_1d_ema200_vol_v1"
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
    
    # 1d data for Camarilla pivots and EMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot point
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Calculate Camarilla levels
    range_1d = high_1d - low_1d
    r3_1d = close_1d + range_1d * 1.1 / 2
    r4_1d = close_1d + range_1d * 1.1
    s3_1d = close_1d - range_1d * 1.1 / 2
    s4_1d = close_1d - range_1d * 1.1
    
    # Align Camarilla levels to 6s timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 1d EMA200 trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 6s volume average for confirmation
    df_6s = get_htf_data(prices, '6h')
    if len(df_6s) < 20:
        return np.zeros(n)
    
    volume_6s = df_6s['volume'].values
    volume_ma_6s = pd.Series(volume_6s).rolling(window=20, min_periods=20).mean().values
    volume_ma_6s_aligned = align_htf_to_ltf(prices, df_6s, volume_ma_6s)
    
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
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_200_aligned[i]) or np.isnan(volume_ma_6s_aligned[i]) or 
            np.isnan(atr[i])):
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
            # Exit: trend reverses (price below EMA200) or retests R3 level
            elif close[i] < ema_200_aligned[i] or close[i] < r3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: trend reverses (price above EMA200) or retests S3 level
            elif close[i] > ema_200_aligned[i] or close[i] > s3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend alignment
            # Long: price breaks above R4 level, price above EMA200 (uptrend), volume spike
            if (close[i] > r4_aligned[i] and
                close[i] > ema_200_aligned[i] and
                volume[i] > 1.5 * volume_ma_6s_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below S4 level, price below EMA200 (downtrend), volume spike
            elif (close[i] < s4_aligned[i] and
                  close[i] < ema_200_aligned[i] and
                  volume[i] > 1.5 * volume_ma_6s_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals