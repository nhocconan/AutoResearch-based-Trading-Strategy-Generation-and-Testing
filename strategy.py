#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal Breakout with Weekly Trend and Volume Confirmation
# Enters long when price breaks above recent bearish fractal high with weekly uptrend and volume > 1.5x average.
# Enters short when price breaks below recent bullish fractal low with weekly downtrend and volume > 1.5x average.
# Exits when price returns to the opposite fractal level.
# Williams Fractals identify key support/resistance levels, weekly trend filter avoids counter-trend trades.
# Volume confirmation ensures institutional participation. Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_WilliamsFractal_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # === Williams Fractals (5-bar window) ===
    high = df_1d['high'].values
    low = df_1d['low'].values
    
    bearish_fractal = np.full(len(high), np.nan)
    bullish_fractal = np.full(len(high), np.nan)
    
    # Calculate fractals: need 2 bars on each side
    for i in range(2, len(high) - 2):
        # Bearish fractal: high[i] is highest among 5 bars
        if (high[i] >= high[i-1] and high[i] >= high[i-2] and 
            high[i] >= high[i+1] and high[i] >= high[i+2]):
            bearish_fractal[i] = high[i]
        # Bullish fractal: low[i] is lowest among 5 bars
        if (low[i] <= low[i-1] and low[i] <= low[i-2] and 
            low[i] <= low[i+1] and low[i] <= low[i+2]):
            bullish_fractal[i] = low[i]
    
    # Williams fractals need 2 extra bars for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Forward fill to get most recent fractal levels
    bearish_fractal_ffill = pd.Series(bearish_fractal_aligned).ffill().values
    bullish_fractal_ffill = pd.Series(bullish_fractal_aligned).ffill().values
    
    # === Weekly EMA20 for trend filter ===
    weekly_close = df_1w['close'].values
    ema_20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    # === Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Get values
        close_val = prices['close'].iloc[i]
        bear_fractal = bearish_fractal_ffill[i]
        bull_fractal = bullish_fractal_ffill[i]
        ema_val = ema_20_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(bear_fractal) or np.isnan(bull_fractal) or np.isnan(ema_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above recent bearish fractal, weekly uptrend, volume confirmation
            if close_val > bear_fractal and close_val > ema_val and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below recent bullish fractal, weekly downtrend, volume confirmation
            elif close_val < bull_fractal and close_val < ema_val and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to bullish fractal level or trend breaks
            if close_val <= bull_fractal or close_val <= ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to bearish fractal level or trend breaks
            if close_val >= bear_fractal or close_val >= ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals