#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour price action combined with daily pivot levels and volume confirmation
# Uses daily Camarilla pivot levels (R3/S3 for reversals, R4/S4 for breakouts) to identify institutional levels
# Long when price crosses above S3 with rejection (close > open) and volume > 1.5x average
# Short when price crosses below R3 with rejection (close < open) and volume > 1.5x average
# Exit when price reaches opposite pivot level (S4 for longs, R4 for shorts) or reverses at R3/S3
# Uses 6-hour ATR for dynamic stoploss at 2.5x ATR
# Position size: 0.25 (25% of capital)
# Target: 100-200 total trades over 4 years (25-50/year)

name = "6h_camarilla_pivot_volume_v1"
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
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # 6h data for price action and ATR
    df_6h = prices.copy()
    
    # Daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Resistance levels
    r1 = close_1d + (range_1d * 1.0 / 12.0)
    r2 = close_1d + (range_1d * 2.0 / 12.0)
    r3 = close_1d + (range_1d * 3.0 / 12.0)
    r4 = close_1d + (range_1d * 4.0 / 12.0)  # Breakout level
    
    # Support levels
    s1 = close_1d - (range_1d * 1.0 / 12.0)
    s2 = close_1d - (range_1d * 2.0 / 12.0)
    s3 = close_1d - (range_1d * 3.0 / 12.0)
    s4 = close_1d - (range_1d * 4.0 / 12.0)  # Breakdown level
    
    # Align pivot levels to 6h timeframe (use previous day's levels)
    r3_1d = align_htf_to_ltf(prices, df_1d, r3)
    r4_1d = align_htf_to_ltf(prices, df_1d, r4)
    s3_1d = align_htf_to_ltf(prices, df_1d, s3)
    s4_1d = align_htf_to_ltf(prices, df_1d, s4)
    
    # 6-hour average volume for confirmation
    volume_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
        if (np.isnan(r3_1d[i]) or np.isnan(r4_1d[i]) or 
            np.isnan(s3_1d[i]) or np.isnan(s4_1d[i]) or 
            np.isnan(volume_ma_6h[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Take profit at S4 (breakdown level) or reverse at R3
            elif close[i] >= s4_1d[i] or close[i] <= r3_1d[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Take profit at R4 (breakout level) or reverse at S3
            elif close[i] <= r4_1d[i] or close[i] >= s3_1d[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            # Long: price crosses above S3 with bullish candle and volume spike
            if (close[i] > s3_1d[i] and open_price[i] <= s3_1d[i] and  # crossed above S3
                close[i] > open_price[i] and                           # bullish candle
                volume[i] > 1.5 * volume_ma_6h[i]):                    # volume confirmation
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price crosses below R3 with bearish candle and volume spike
            elif (close[i] < r3_1d[i] and open_price[i] >= r3_1d[i] and  # crossed below R3
                  close[i] < open_price[i] and                           # bearish candle
                  volume[i] > 1.5 * volume_ma_6h[i]):                    # volume confirmation
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals