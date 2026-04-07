#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Donchian(20) breakout with weekly pivot direction and volume confirmation
# Long when price breaks above 20-period Donchian high + weekly pivot support level acts as support + volume > 1.5x 20-period average
# Short when price breaks below 20-period Donchian low + weekly pivot resistance level acts as resistance + volume > 1.5x 20-period average
# Exit when price crosses 6-period EMA in opposite direction
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses weekly pivot points for structural bias and volume for confirmation
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_donchian20_1w_pivot_vol_v1"
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
    
    # 1-week data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # Weekly pivot: P = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Support 1: S1 = (2 * P) - H
    s1_1w = (2 * pivot_1w) - high_1w
    # Resistance 1: R1 = (2 * P) - L
    r1_1w = (2 * pivot_1w) - low_1w
    # Support 2: S2 = P - (H - L)
    s2_1w = pivot_1w - (high_1w - low_1w)
    # Resistance 2: R2 = P + (H - L)
    r2_1w = pivot_1w + (high_1w - low_1w)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    
    # 20-period Donchian channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 6-period EMA for exit
    ema_6 = pd.Series(close).ewm(span=6, adjust=False, min_periods=6).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 20-period volume average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_6[i]) or np.isnan(atr[i]) or np.isnan(volume_ma[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i])):
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
            # Exit: price crosses below 6-period EMA
            elif close[i] < ema_6[i]:
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
            # Exit: price crosses above 6-period EMA
            elif close[i] > ema_6[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Volume filter: volume > 1.5x 20-period average
            volume_filter = volume[i] > 1.5 * volume_ma[i]
            
            # Long: price breaks above Donchian high + price above weekly S1 + volume filter
            if (close[i] > highest_high[i] and 
                close[i] > s1_aligned[i] and 
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low + price below weekly R1 + volume filter
            elif (close[i] < lowest_low[i] and 
                  close[i] < r1_aligned[i] and 
                  volume_filter):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals