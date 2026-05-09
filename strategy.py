#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with volume confirmation and 1d EMA100 trend filter.
# Uses wider breakout levels (R4/S4) to capture stronger moves with fewer trades.
# 1d EMA100 filters for strong trend direction, avoiding choppy markets.
# Volume > 2x 20-period EMA ensures high conviction moves.
# Designed for low trade frequency (<30/year) to minimize fee drag in BTC/ETH markets.
name = "4h_Camarilla_R4S4_Breakout_1dEMA100_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA100 trend and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: Range = High - Low
    range_1d = high_1d - low_1d
    r4 = close_1d + (range_1d * 1.5000)  # R4 level
    s4 = close_1d - (range_1d * 1.5000)  # S4 level
    
    # Use previous day's levels (shift by 1 to avoid look-ahead)
    r4_shifted = np.roll(r4, 1)
    s4_shifted = np.roll(s4, 1)
    r4_shifted[0] = np.nan
    s4_shifted[0] = np.nan
    
    # Align to 4h timeframe
    r4_4h = align_htf_to_ltf(prices, df_1d, r4_shifted)
    s4_4h = align_htf_to_ltf(prices, df_1d, s4_shifted)
    
    # 1d EMA100 trend filter
    ema_100_1d = pd.Series(df_1d['close'].values).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    # Volume spike filter: volume > 2x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(r4_4h[i]) or np.isnan(s4_4h[i]) or
            np.isnan(ema_100_1d_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above R4 with volume spike and above 1d EMA100
            if (price > r4_4h[i] and vol_spike[i] and price > ema_100_1d_aligned[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below S4 with volume spike and below 1d EMA100
            elif (price < s4_4h[i] and vol_spike[i] and price < ema_100_1d_aligned[i]):
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below S4 (mean reversion to support)
            if price < s4_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit short: price rises back above R4 (mean reversion to resistance)
            if price > r4_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals