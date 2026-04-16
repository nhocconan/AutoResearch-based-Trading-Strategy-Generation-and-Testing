#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 12h trend filter and volume confirmation
# Fractals provide structural support/resistance levels. Breakouts above/below recent
# fractal levels with volume and trend alignment capture momentum moves.
# Works in bull/bear by following institutional interest at key turning points.
# Target: 20-50 trades/year (80-200 over 4 years) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h data (HTF for trend and fractals) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # === Williams Fractals on 12h (5-bar pattern: high/low surrounded by 2 lower/higher) ===
    # Bearish fractal: high[i] is highest of [i-2, i-1, i, i+1, i+2]
    # Bullish fractal: low[i] is lowest of [i-2, i-1, i, i+1, i+2]
    n_12h = len(high_12h)
    bearish_fractal = np.full(n_12h, np.nan)
    bullish_fractal = np.full(n_12h, np.nan)
    
    for i in range(2, n_12h - 2):
        # Bearish fractal (peak)
        if (high_12h[i] > high_12h[i-2] and high_12h[i] > high_12h[i-1] and
            high_12h[i] > high_12h[i+1] and high_12h[i] > high_12h[i+2]):
            bearish_fractal[i] = high_12h[i]
        # Bullish fractal (trough)
        if (low_12h[i] < low_12h[i-2] and low_12h[i] < low_12h[i-1] and
            low_12h[i] < low_12h[i+1] and low_12h[i] < low_12h[i+2]):
            bullish_fractal[i] = low_12h[i]
    
    # Need 2-bar confirmation after the fractal bar (Williams requirement)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_12h, bullish_fractal, additional_delay_bars=2)
    
    # === 12h EMA for trend direction (34 period) ===
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # === 12h volume ratio for confirmation ===
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = volume_12h / vol_ma_20_12h
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    signals = np.zeros(n)
    
    # Warmup: enough for EMA and fractal calculations
    warmup = 50
    
    # Track position and entry price
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(vol_ratio_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_trend = ema_12h_aligned[i]
        vol_ratio_val = vol_ratio_12h_aligned[i]
        resistance = bearish_fractal_aligned[i]  # Recent swing high
        support = bullish_fractal_aligned[i]     # Recent swing low
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: close below support or 2x ATR stop (using price action)
            if price < support or (i > 0 and price < close[i-1] - 0.5 * (high[i-1] - low[i-1])):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: close above resistance or 2x ATR stop
            if price > resistance or (i > 0 and price > close[i-1] + 0.5 * (high[i-1] - low[i-1])):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price breaks above resistance with volume and uptrend
            if (not np.isnan(resistance) and price > resistance and 
                price > ema_trend and vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            # SHORT: price breaks below support with volume and downtrend
            elif (not np.isnan(support) and price < support and 
                  price < ema_trend and vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
                entry_price = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Fractal_Breakout_Trend_Volume"
timeframe = "6h"
leverage = 1.0