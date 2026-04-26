#!/usr/bin/env python3
"""
4h_Keltner_Breakout_1dTrend_VolumeFilter
Hypothesis: 4h breakout above/below Keltner Channel (EMA20 ± 2*ATR10) in direction of 1d EMA50 trend, confirmed by volume > 1.5x 20-bar MA. Keltner Channels adapt to volatility, providing dynamic support/resistance. Trend filter ensures alignment with higher timeframe momentum. Volume confirmation reduces false breakouts. Designed for 20-40 trades/year (80-160 total over 4 years) to avoid fee drag. Works in both bull and bear markets by following the 1d trend while using volatility-based channels for precise entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h EMA20 for Keltner middle line
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 4h ATR10 for Keltner width
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr1 = np.maximum(tr1, np.abs(low[1:] - close[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])  # same length as close
    atr10 = pd.Series(tr1).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner Channels: EMA20 ± 2*ATR10
    keltner_upper = ema_20 + 2.0 * atr10
    keltner_lower = ema_20 - 2.0 * atr10
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size
    
    # Warmup: max of calculations (20 for EMA/vol, 10 for ATR, 50 for 1d EMA)
    start_idx = max(20, 10, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(keltner_upper[i]) or 
            np.isnan(keltner_lower[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_1d_aligned[i]
        upper = keltner_upper[i]
        lower = keltner_lower[i]
        vol_ok = volume_filter[i]
        
        # Determine 1d trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_1d = close_val > ema_50_val
        bearish_1d = close_val < ema_50_val
        
        # Entry conditions: breakout of Keltner Channel in trend direction with volume filter
        long_entry = (close_val > upper) and bullish_1d and vol_ok
        short_entry = (close_val < lower) and bearish_1d and vol_ok
        
        # Exit conditions: opposite channel touch (or trend reversal)
        exit_long = (close_val < lower) or not bullish_1d
        exit_short = (close_val > upper) or not bearish_1d
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Keltner_Breakout_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0