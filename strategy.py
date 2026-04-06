#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d EMA200 + Volume
# Williams %R measures overbought/oversold conditions.
# In strong trends, it can remain extreme; we use 1d EMA200 to filter trend direction.
# Long: Williams %R < -80 (oversold) + price > 1d EMA200 (uptrend) + volume spike
# Short: Williams %R > -20 (overbought) + price < 1d EMA200 (downtrend) + volume spike
# Exit when Williams %R reverts to -50 or trend fails.
# Uses volume confirmation to avoid false reversals. Target: 75-200 trades over 4 years.

name = "6h_williamsr_1d_ema200_vol_v1"
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
    
    # 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # EMA200 calculation
    ema200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Align 1d EMA200 to 6h timeframe
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r[highest_high == lowest_low] = -50
    
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
    
    for i in range(14, n):  # Start after Williams %R warmup
        # Skip if required data not available
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(williams_r[i]) or 
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
            # Exit: Williams %R reverts to -50 or trend fails
            elif williams_r[i] >= -50 or close[i] < ema200_1d_aligned[i]:
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
            # Exit: Williams %R reverts to -50 or trend fails
            elif williams_r[i] <= -50 or close[i] > ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            # Long: Williams %R < -80 (oversold) + uptrend + volume spike
            if (williams_r[i] < -80 and 
                close[i] > ema200_1d_aligned[i] and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Williams %R > -20 (overbought) + downtrend + volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < ema200_1d_aligned[i] and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals