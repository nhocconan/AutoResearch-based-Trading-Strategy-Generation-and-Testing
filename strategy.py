#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1d trend filter
# - Uses 1d Williams Fractals to identify key swing highs/lows
# - Enters long when price breaks above most recent bullish fractal (with 1d uptrend filter)
# - Enters short when price breaks below most recent bearish fractal (with 1d downtrend filter)
# - Uses 1d EMA(50) as trend filter: only long when price > EMA50, short when price < EMA50
# - Target: 12-25 trades/year on 6h timeframe (50-100 total over 4 years) to avoid fee drag
# - Williams Fractals provide natural support/resistance levels that work in both trending and ranging markets

name = "6h_1d_williams_fractal_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d Williams Fractals
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams Fractal: need 5 points (n-2, n-1, n, n+1, n+2)
    # Bullish fractal: lowest low in middle with higher lows on both sides
    # Bearish fractal: highest high in middle with lower highs on both sides
    bullish_fractal = np.full(len(low_1d), np.nan)
    bearish_fractal = np.full(len(high_1d), np.nan)
    
    for i in range(2, len(low_1d) - 2):
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
    
    # Forward fill to get most recent fractal levels
    bullish_fractal = pd.Series(bullish_fractal).ffill().values
    bearish_fractal = pd.Series(bearish_fractal).ffill().values
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h timeframe
    # Williams fractals need extra delay for confirmation (2 additional 1d bars)
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Pre-compute 6h ATR(14) for stoploss
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(atr[i]) or atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # Exit conditions: stoploss or trend reversal
            if close[i] < highest_high_since_entry - 2.5 * atr[i]:  # ATR stop
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            elif close[i] < ema_50_aligned[i]:  # Trend filter exit (price below EMA50)
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # Exit conditions: stoploss or trend reversal
            if close[i] > lowest_low_since_entry + 2.5 * atr[i]:  # ATR stop
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            elif close[i] > ema_50_aligned[i]:  # Trend filter exit (price above EMA50)
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for breakout entries with trend filter
            if close[i] > bullish_aligned[i] and close[i] > ema_50_aligned[i]:  # Break above bullish fractal with uptrend
                position = 1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = 0.25
            elif close[i] < bearish_aligned[i] and close[i] < ema_50_aligned[i]:  # Break below bearish fractal with downtrend
                position = -1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = -0.25
    
    return signals