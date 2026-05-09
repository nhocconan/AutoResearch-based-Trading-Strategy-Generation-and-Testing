#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with volume confirmation and 12h EMA200 trend filter.
# Uses ultra-wide breakout levels (R4/S4) for very low-frequency, high-conviction trades.
# 12h EMA200 filters for long-term trend direction to avoid counter-trend entries.
# Volume > 2.0x 50-period EMA ensures strong institutional participation.
# Designed for extreme selectivity (<20 trades/year) to minimize fee drag in choppy markets.
name = "4h_Camarilla_R4S4_Breakout_12hEMA200_Volume"
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
    
    # 12h data for EMA200 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 200:
        return np.zeros(n)
    
    # Daily data for Camarilla levels (wider R4/S4 levels)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla
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
    
    # 12h EMA200 trend filter
    ema_200_12h = pd.Series(df_12h['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    # Volume spike filter: volume > 2.0x 50-period EMA
    vol_ema50 = pd.Series(volume).ewm(span=50, adjust=False, min_periods=50).mean().values
    vol_spike = volume > (2.0 * vol_ema50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(r4_4h[i]) or np.isnan(s4_4h[i]) or
            np.isnan(ema_200_12h_aligned[i]) or np.isnan(vol_ema50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above R4 with volume spike and above 12h EMA200
            if (price > r4_4h[i] and vol_spike[i] and price > ema_200_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 with volume spike and below 12h EMA200
            elif (price < s4_4h[i] and vol_spike[i] and price < ema_200_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below S4 (mean reversion to support)
            if price < s4_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above R4 (mean reversion to resistance)
            if price > r4_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals