#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal breakout with 1d ATR volatility filter and 1w trend filter
# Long when price breaks above recent bearish fractal with expanding volatility and weekly uptrend
# Short when price breaks below recent bullish fractal with expanding volatility and weekly downtrend
# Williams Fractals identify key support/resistance levels, ATR filters for volatility expansion,
# Weekly trend ensures alignment with higher timeframe momentum
# Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "12h_WilliamsFractal_Breakout_ATRVol_1wTrend"
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
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend filter
    weekly_close = df_1w['close'].values
    ema34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Get daily data for ATR and fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    tr3 = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    tr3 = np.abs(np.roll(close_1d, 1) - close_1d)
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate Williams Fractals on daily data
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bearish_fractal = np.zeros(len(high_1d), dtype=bool)
    bullish_fractal = np.zeros(len(low_1d), dtype=bool)
    
    for i in range(2, len(high_1d)-2):
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i] < high_1d[i-1] and
            high_1d[i-3] < high_1d[i-1] and
            high_1d[i+1] < high_1d[i-1]):
            bearish_fractal[i-1] = True
            
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i] > low_1d[i-1] and
            low_1d[i-3] > low_1d[i-1] and
            low_1d[i+1] > low_1d[i-1]):
            bullish_fractal[i-1] = True
    
    # Store actual price levels of fractals
    bearish_level = np.where(bearish_fractal, high_1d, np.nan)
    bullish_level = np.where(bullish_fractal, low_1d, np.nan)
    
    # Forward fill to get the most recent fractal level
    bearish_level = pd.Series(bearish_level).ffill().values
    bullish_level = pd.Series(bullish_level).ffill().values
    
    # Align fractal levels to 12h timeframe
    bearish_level_aligned = align_htf_to_ltf(prices, df_1d, bearish_level)
    bullish_level_aligned = align_htf_to_ltf(prices, df_1d, bullish_level)
    
    # Calculate ATR-based threshold for breakout confirmation
    atr_threshold = atr_1d_aligned * 0.5  # Half ATR for breakout confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(close[i]) or np.isnan(bearish_level_aligned[i]) or 
            np.isnan(bullish_level_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        bear_level = bearish_level_aligned[i]
        bull_level = bullish_level_aligned[i]
        atr_val = atr_1d_aligned[i]
        ema34_1w_val = ema34_1w_aligned[i]
        threshold = atr_threshold[i]
        
        if position == 0:
            # Enter long: price breaks above recent bearish fractal with volatility expansion
            if not np.isnan(bear_level) and price > bear_level + threshold and ema34_1w_val > 0:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below recent bullish fractal with volatility expansion
            elif not np.isnan(bull_level) and price < bull_level - threshold and ema34_1w_val < 0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below recent bullish fractal or weekly trend down
            if not np.isnan(bull_level) and price < bull_level - threshold or ema34_1w_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above recent bearish fractal or weekly trend up
            if not np.isnan(bear_level) and price > bear_level + threshold or ema34_1w_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals