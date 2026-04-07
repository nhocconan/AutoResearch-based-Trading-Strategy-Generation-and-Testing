#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour ATR-based breakout with daily trend filter and volume confirmation
# ATR breakouts capture volatility expansion in both trending and ranging markets.
# Daily trend filter ensures trades align with higher timeframe direction.
# Volume confirmation filters low-probability breakouts. Designed for low frequency.
# Works in bull markets (breakouts above resistance in uptrend) and bear markets (breakdowns below support in downtrend).

name = "4h_atr_breakout_1d_trend_volume_v1"
timeframe = "4h"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA20 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema20_1d = close_1d.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Calculate ATR (14) for breakout threshold
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate rolling max/min for breakout levels (20 periods)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after ATR and rolling window warmup
        # Skip if required data not available
        if (np.isnan(ema20_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Daily trend: close above/below daily EMA20
        daily_uptrend = close[i] > ema20_1d_aligned[i]
        daily_downtrend = close[i] < ema20_1d_aligned[i]
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Breakout conditions with ATR filter
        bullish_breakout = close[i] > highest_high[i] + 0.5 * atr[i]
        bearish_breakout = close[i] < lowest_low[i] - 0.5 * atr[i]
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit if daily trend turns down or price closes below 20-period low
            if not daily_uptrend or close[i] < lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit if daily trend turns up or price closes above 20-period high
            if not daily_downtrend or close[i] > highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: daily uptrend + bullish breakout + volume confirmation
            if daily_uptrend and bullish_breakout and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: daily downtrend + bearish breakout + volume confirmation
            elif daily_downtrend and bearish_breakout and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals