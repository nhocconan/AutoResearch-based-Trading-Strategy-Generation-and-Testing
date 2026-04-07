#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla pivot strategy using daily pivot levels
# Uses daily Camarilla levels (calculated from previous day's OHLC) for mean reversion and breakout signals:
# - Long when price touches S3 level with bullish engulfing candle AND closes above S2
# - Short when price touches R3 level with bearish engulfing candle AND closes below R2
# - Exit when price reaches opposite H3/L3 level or reverses at H4/L4
# - Designed for low frequency (target: 15-30 trades/year) by requiring specific price action at key levels
# - Camarilla levels work well in ranging markets (2022-2024) and capture breakouts in trending markets (2021, 2025+)

name = "6h_camarilla_pivot_1d_mean_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    open_prices = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for today's Camarilla levels
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan  # First day has no previous
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla multipliers
    H3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    L3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    H4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    L4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align daily levels to 6h timeframe
    H3_6h = align_htf_to_ltf(prices, df_1d, H3)
    L3_6h = align_htf_to_ltf(prices, df_1d, L3)
    H4_6h = align_htf_to_ltf(prices, df_1d, H4)
    L4_6h = align_htf_to_ltf(prices, df_1d, L4)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(1, n):  # Start from 1 to access i-1 for candle patterns
        # Skip if Camarilla levels not available
        if (np.isnan(H3_6h[i]) or np.isnan(L3_6h[i]) or 
            np.isnan(H4_6h[i]) or np.isnan(L4_6h[i])):
            signals[i] = 0.0
            continue
        
        # Bullish engulfing pattern: current candle engulfs previous bearish candle
        bullish_engulf = (close[i] > open_prices[i] and  # Current bullish
                         open_prices[i-1] > close[i-1] and  # Previous bearish
                         close[i] > open_prices[i-1] and  # Current close > previous open
                         open_prices[i] < close[i-1])     # Current open < previous close
        
        # Bearish engulfing pattern: current candle engulfs previous bullish candle
        bearish_engulf = (close[i] < open_prices[i] and  # Current bearish
                         open_prices[i-1] < close[i-1] and  # Previous bullish
                         close[i] < open_prices[i-1] and  # Current close < previous open
                         open_prices[i] > close[i-1])     # Current open > previous close
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit at H3 (take profit) or if price goes below L4 (stop)
            if close[i] >= H3_6h[i] or close[i] <= L4_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit at L3 (take profit) or if price goes above H4 (stop)
            if close[i] <= L3_6h[i] or close[i] >= H4_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price touches L3 with bullish engulfing
            if (bullish_engulf and 
                low[i] <= L3_6h[i] * 1.001 and  # Touched or went slightly below L3
                close[i] > L3_6h[i]):  # Closed back above L3
                position = 1
                signals[i] = 0.25
            # Short: price touches H3 with bearish engulfing
            elif (bearish_engulf and 
                  high[i] >= H3_6h[i] * 0.999 and  # Touched or went slightly above H3
                  close[i] < H3_6h[i]):  # Closed back below H3
                position = -1
                signals[i] = -0.25
    
    return signals