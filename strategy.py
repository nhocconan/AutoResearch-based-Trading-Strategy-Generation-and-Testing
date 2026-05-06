#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d ATR-based volatility regime filter + 1w EMA200 trend filter + price action breakout
# Long when price breaks above 12h Donchian(20) high AND 1d ATR(14)/ATR(50) > 1.2 (expanding volatility) AND 1w EMA200 > EMA200 previous (uptrend)
# Short when price breaks below 12h Donchian(20) low AND 1d ATR(14)/ATR(50) > 1.2 (expanding volatility) AND 1w EMA200 < EMA200 previous (downtrend)
# Exit when price crosses 12h EMA20 (mean reversion to short-term trend)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# ATR regime filter identifies true breakouts vs false signals in ranging markets
# 1w EMA200 trend filter ensures we trade with the dominant weekly trend
# Donchian breakout provides clear entry/exit levels with built-in structure
# Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend) markets

name = "12h_ATRRegime_DonchianBreakout_1wEMA200_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need at least 50 completed 1d bars for ATR(50)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) and ATR(50) for volatility regime
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], tr2])
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = np.where(atr_50 != 0, atr_14 / atr_50, 1.0)
    
    # Align 1d ATR ratio to 12h timeframe (wait for completed 1d bar)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Get 1w data ONCE before loop for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:  # Need at least 200 completed weekly bars for EMA200
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA200
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate 12h Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h EMA20 for exit signal
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout above, ATR regime > 1.2 (expanding vol), 1w EMA200 rising, in session
            if (close[i] > highest_high_20[i] and 
                atr_ratio_aligned[i] > 1.2 and 
                ema_200_1w_aligned[i] > ema_200_1w_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown below, ATR regime > 1.2 (expanding vol), 1w EMA200 falling, in session
            elif (close[i] < lowest_low_20[i] and 
                  atr_ratio_aligned[i] > 1.2 and 
                  ema_200_1w_aligned[i] < ema_200_1w_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 12h EMA20 (mean reversion)
            if close[i] < ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 12h EMA20 (mean reversion)
            if close[i] > ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals