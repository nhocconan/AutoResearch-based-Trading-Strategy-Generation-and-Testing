#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and session filter
# - Long when price breaks above H3 pivot level AND 4h close > 4h open (bullish 4h candle) AND hour in [8,20) UTC
# - Short when price breaks below L3 pivot level AND 4h close < 4h open (bearish 4h candle) AND hour in [8,20) UTC
# - Exit when price returns to the 4h mid-point (pivot) level
# - Uses discrete position sizing 0.20 to limit fee churn
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)
# - Camarilla pivots provide precise intraday support/resistance levels
# - 4h trend filter ensures we trade with the higher timeframe momentum
# - Session filter (08-20 UTC) avoids low-volume Asian session noise
# - Works in both bull and bear markets by following 4h trend direction

name = "1h_4h_camarilla_pivot_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Pre-compute 1h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Pre-compute 1h Camarilla pivot levels (based on previous bar)
    # H4 = close + 1.5*(high-low), H3 = close + 1.25*(high-low), etc.
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # First bar uses current values
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    # Calculate pivot range
    pivot_range = prev_high - prev_low
    
    # Camarilla levels
    h5 = prev_close + 1.625 * pivot_range
    h4 = prev_close + 1.500 * pivot_range
    h3 = prev_close + 1.250 * pivot_range
    l3 = prev_close - 1.250 * pivot_range
    l4 = prev_close - 1.500 * pivot_range
    l5 = prev_close - 1.625 * pivot_range
    
    # Pivot point (mid-point)
    pivot = (prev_high + prev_low + prev_close) / 3.0
    
    # Pre-compute 4h trend (bullish/bearish candle)
    # We need 4h open, high, low, close - but we only have aligned close
    # So we'll use: bullish if 4h close > 4h open
    # To get 4h open, we need to load the full 4h data
    open_4h = df_4h['open'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h bullish candle: close > open
    bullish_4h = close_4h > open_4h
    bearish_4h = close_4h < open_4h
    
    # Align 4h indicators to 1h timeframe
    bullish_4h_aligned = align_htf_to_ltf(prices, df_4h, bullish_4h)
    bearish_4h_aligned = align_htf_to_ltf(prices, df_4h, bearish_4h)
    pivot_aligned = align_htf_to_ltf(prices, df_4h, pivot)
    
    # Pre-compute session filter (08-20 UTC)
    # prices.index is already DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours < 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(h3[i]) or np.isnan(l3[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(bullish_4h_aligned[i]) or np.isnan(bearish_4h_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above H3 AND 4h bullish AND in session
            if (close[i] > h3[i] and 
                bullish_4h_aligned[i] and 
                in_session[i]):
                position = 1
                signals[i] = 0.20
            # Short conditions: price breaks below L3 AND 4h bearish AND in session
            elif (close[i] < l3[i] and 
                  bearish_4h_aligned[i] and 
                  in_session[i]):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to pivot level
            exit_long = (position == 1 and close[i] < pivot_aligned[i])
            exit_short = (position == -1 and close[i] > pivot_aligned[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
    
    return signals