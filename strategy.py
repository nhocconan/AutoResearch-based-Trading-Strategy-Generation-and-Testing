#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Keltner Channel breakout with 12h trend filter and volume confirmation.
# Long when price closes above upper KC (EMA20 + 2*ATR) AND 12h EMA50 rising AND volume > 1.5x 20-period average.
# Short when price closes below lower KC (EMA20 - 2*ATR) AND 12h EMA50 falling AND volume > 1.5x 20-period average.
# Exit when price closes back inside KC (between upper and lower bands).
# Keltner Channels adapt to volatility, reducing false breakouts in choppy markets.
# 12h EMA50 ensures alignment with higher timeframe trend.
# Volume confirmation filters out low-participation moves.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled risk.

name = "6h_Keltner_20_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for Keltner calculation and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate EMA20 and ATR(20) for Keltner Channels
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    tr_12h = np.maximum(
        high_12h[1:] - low_12h[1:],
        np.maximum(
            np.abs(high_12h[1:] - close_12h[:-1]),
            np.abs(low_12h[1:] - close_12h[:-1])
        )
    )
    tr_12h = np.concatenate([[np.nan], tr_12h])  # align length
    atr20_12h = pd.Series(tr_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channels
    kc_upper = ema20_12h + 2 * atr20_12h
    kc_lower = ema20_12h - 2 * atr20_12h
    
    # Align KC levels to 6h timeframe
    kc_upper_aligned = align_htf_to_ltf(prices, df_12h, kc_upper)
    kc_lower_aligned = align_htf_to_ltf(prices, df_12h, kc_lower)
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 12h EMA50 direction
    ema50_rising = np.zeros_like(ema50_12h_aligned, dtype=bool)
    ema50_falling = np.zeros_like(ema50_12h_aligned, dtype=bool)
    ema50_rising[1:] = ema50_12h_aligned[1:] > ema50_12h_aligned[:-1]
    ema50_falling[1:] = ema50_12h_aligned[1:] < ema50_12h_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average (on 6h data)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 20)  # Sufficient warmup for EMA50 and KC
    
    for i in range(start_idx, n):
        if (np.isnan(kc_upper_aligned[i]) or np.isnan(kc_lower_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(ema50_rising[i]) or 
            np.isnan(ema50_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: close above upper KC, 12h EMA50 rising, volume filter
            long_cond = (close[i] > kc_upper_aligned[i]) and ema50_rising[i] and volume_filter[i]
            # Short conditions: close below lower KC, 12h EMA50 falling, volume filter
            short_cond = (close[i] < kc_lower_aligned[i]) and ema50_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close back below lower KC
            if close[i] < kc_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close back above upper KC
            if close[i] > kc_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals