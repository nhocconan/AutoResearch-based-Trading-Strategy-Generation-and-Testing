#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h ATR-based volatility breakout with 1d EMA50 trend filter
# Long when price breaks above 6h Donchian(20) high AND 12h ATR(14) > 1.5 * 12h ATR(50) AND close > 1d EMA50
# Short when price breaks below 6h Donchian(20) low AND 12h ATR(14) > 1.5 * 12h ATR(50) AND close < 1d EMA50
# Exit when price crosses 6h EMA20 in opposite direction
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# ATR expansion identifies genuine breakouts while filtering low-volatility false signals
# 1d EMA50 ensures alignment with higher timeframe trend to avoid counter-trend trades
# Donchian channels provide objective breakout levels with clear risk definition

name = "6h_ATR_VolatilityBreakout_12h_ATR_Ratio_1dEMA50"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data ONCE before loop for ATR calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough for ATR(50)
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate True Range for 12h
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate ATR(14) and ATR(50) for 12h
    atr14_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr50_12h = pd.Series(tr_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align ATR ratios to 6h timeframe (wait for completed 12h bar)
    atr_ratio_12h = atr14_12h / atr50_12h
    atr_ratio_aligned = align_htf_to_ltf(prices, df_12h, atr_ratio_12h)
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA50
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 6h Donchian(20) channels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h EMA20 for exit
    close_series = pd.Series(close)
    ema20_6h = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(ema20_6h[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout above, ATR expansion, above 1d EMA50
            if (close[i] > highest_high_20[i] and atr_ratio_aligned[i] > 1.5 and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown below, ATR expansion, below 1d EMA50
            elif (close[i] < lowest_low_20[i] and atr_ratio_aligned[i] > 1.5 and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 6h EMA20
            if close[i] < ema20_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 6h EMA20
            if close[i] > ema20_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals