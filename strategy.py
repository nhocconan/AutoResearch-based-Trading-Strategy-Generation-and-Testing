#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla pivot levels from daily data with volume confirmation
# Uses daily Camarilla pivot levels (R3/S3, R4/S4) for mean reversion and breakout signals
# Long at S3 with stop at S4, short at R3 with stop at R4, requires volume > 1.5x average
# Trend filter: price above/below 20-period EMA on 6h to avoid counter-trend trades
# Designed for low trade frequency (target: 75-150 total trades over 4 years)
# Works in both bull and bear markets by fading extremes and following breakouts

name = "6h_camarilla_1d_vol_trend_v1"
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
    
    # 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels for previous day
    # R4 = Close + 1.5 * (High - Low)
    # R3 = Close + 1.0 * (High - Low)
    # S3 = Close - 1.0 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    camarilla_base = prev_close
    camarilla_range = prev_high - prev_low
    
    r4 = camarilla_base + 1.5 * camarilla_range
    r3 = camarilla_base + 1.0 * camarilla_range
    s3 = camarilla_base - 1.0 * camarilla_range
    s4 = camarilla_base - 1.5 * camarilla_range
    
    # Align Camarilla levels to 6h timeframe (previous day's levels)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 6h EMA20 for trend filter
    ema20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    
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
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema20[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR or break below S4
            if close[i] < entry_price - 2.0 * atr[i] or close[i] < s4_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price reaches R3 or trend turns down
            elif close[i] >= r3_aligned[i] or close[i] < ema20[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR or break above R4
            if close[i] > entry_price + 2.0 * atr[i] or close[i] > r4_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price reaches S3 or trend turns up
            elif close[i] <= s3_aligned[i] or close[i] > ema20[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for mean reversion entries at S3/R3 with volume confirmation
            # Long: price touches S3, volume spike, above EMA20 (uptrend bias)
            if (close[i] <= s3_aligned[i] and
                volume[i] > 1.5 * volume_ma[i] and
                close[i] > ema20[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price touches R3, volume spike, below EMA20 (downtrend bias)
            elif (close[i] >= r3_aligned[i] and
                  volume[i] > 1.5 * volume_ma[i] and
                  close[i] < ema20[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            # Breakout entries: price breaks S4/R4 with volume
            elif (close[i] < s4_aligned[i] and
                  volume[i] > 2.0 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif (close[i] > r4_aligned[i] and
                  volume[i] > 2.0 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals