#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_Pivot_MeanReversion_Trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Previous day's OHLC for pivot calculation ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Previous day's values (shifted by 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_open = np.roll(open_1d, 1)
    
    # Fill first values with current day's values to avoid look-ahead
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    prev_open[0] = open_1d[0]
    
    # Calculate pivot point (standard formula)
    pivot = (prev_high + prev_low + prev_close) / 3.0
    
    # Calculate range
    range_1d = prev_high - prev_low
    
    # Camarilla levels (standard formulas)
    r1 = pivot + (range_1d * 1.1 / 12)
    s1 = pivot - (range_1d * 1.1 / 12)
    r2 = pivot + (range_1d * 1.1 / 6)
    s2 = pivot - (range_1d * 1.1 / 6)
    r3 = pivot + (range_1d * 1.1 / 4)
    s3 = pivot - (range_1d * 1.1 / 4)
    
    # Align levels to 12h timeframe
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    
    # === 12h Moving average for trend filter (20-period EMA) ===
    ema20_12h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # === Volume filter: current volume > 20-period average ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Trend filter: 12h price above/below EMA20 ===
    price_above_ema = close > ema20_12h
    price_below_ema = close < ema20_12h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for EMA20 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or np.isnan(r2_12h[i]) or np.isnan(s2_12h[i]) or
            np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or np.isnan(ema20_12h[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Mean reversion entry: price at S1/S3 with trend filter
            long_cond = (close[i] <= s1_12h[i] and 
                        price_above_ema[i] and  # Only buy in uptrend
                        volume[i] > vol_ma20[i])
            
            short_cond = (close[i] >= r1_12h[i] and 
                         price_below_ema[i] and  # Only sell in downtrend
                         volume[i] > vol_ma20[i])
            
            # Additional entry at S3/R3 for stronger reversals
            long_cond_strong = (close[i] <= s3_12h[i] and 
                               volume[i] > vol_ma20[i])
            
            short_cond_strong = (close[i] >= r3_12h[i] and 
                                volume[i] > vol_ma20[i])
            
            if long_cond or long_cond_strong:
                signals[i] = 0.25
                position = 1
            elif short_cond or short_cond_strong:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price at R1 or stop loss at S2
            if close[i] >= r1_12h[i] or close[i] <= s2_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price at S1 or stop loss at R2
            if close[i] <= s1_12h[i] or close[i] >= r2_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla pivot-based mean reversion strategy for 12h timeframe.
# Enters long at S1/S3 when price is above EMA20 (uptrend) and short at R1/R3 when
# price is below EMA20 (downtrend), with volume confirmation. Exits at opposite
# pivot levels (R1 for longs, S1 for shorts) with stop loss at S2/R2.
# Works in both bull and bear markets by adapting to trend direction.
# Targets 50-150 trades over 4 years (12-37/year) to minimize fee drag.
# Uses discrete sizing (0.25) to reduce churn. Focuses on BTC/ETH via institutional
# pivot levels that act as support/resistance in all market conditions.