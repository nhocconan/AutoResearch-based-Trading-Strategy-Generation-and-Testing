#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1w EMA50 trend filter and volume spike confirmation.
# Williams Fractals identify swing points: bearish fractal (high with lower highs on both sides),
# bullish fractal (low with higher lows on both sides). Breakout above bearish fractal or below bullish fractal
# indicates momentum continuation. 1w EMA50 ensures alignment with weekly trend, volume spike (>2x average) confirms strength.
# Designed to work in both bull (breakouts above fractals in uptrend) and bear (breakdowns below fractals in downtrend).
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_WilliamsFractalBreakout_1wEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 1w close
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams Fractals (5-bar window: 2 left, 2 right)
    # Bearish fractal: high[n] > high[n-2] and high[n] > high[n-1] and high[n] > high[n+1] and high[n] > high[n+2]
    # Bullish fractal: low[n] < low[n-2] and low[n] < low[n-1] and low[n] < low[n+1] and low[n] < low[n+2]
    bearish_fractal = np.full(n, np.nan)
    bullish_fractal = np.full(n, np.nan)
    
    for i in range(2, n-2):
        if (high[i] > high[i-2] and high[i] > high[i-1] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish_fractal[i] = high[i]
        if (low[i] < low[i-2] and low[i] < low[i-1] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            bullish_fractal[i] = low[i]
    
    # Forward fill fractal levels to act as support/resistance until broken
    bearish_fractal_ff = np.where(np.isnan(bearish_fractal), np.nan, bearish_fractal)
    bullish_fractal_ff = np.where(np.isnan(bullish_fractal), np.nan, bullish_fractal)
    
    # Forward fill with previous valid value
    last_bear = np.nan
    last_bull = np.nan
    for i in range(n):
        if not np.isnan(bearish_fractal_ff[i]):
            last_bear = bearish_fractal_ff[i]
        else:
            bearish_fractal_ff[i] = last_bear
            
        if not np.isnan(bullish_fractal_ff[i]):
            last_bull = bullish_fractal_ff[i]
        else:
            bullish_fractal_ff[i] = last_bull
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need 50 periods for 1w EMA50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(bearish_fractal_ff[i]) or np.isnan(bullish_fractal_ff[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_1w = ema_50_1w_aligned[i]
        bear_fractal = bearish_fractal_ff[i]
        bull_fractal = bullish_fractal_ff[i]
        vol = volume[i]
        
        # Calculate 20-period volume average for spike detection
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
        else:
            vol_ma = np.mean(volume[:i]) if i > 0 else volume[i]
        
        if position == 0:
            # Enter long: Price breaks above bearish fractal AND price > 1w EMA50 (uptrend) AND volume > 2x average
            if close[i] > bear_fractal and close[i] > ema_1w and vol > 2.0 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below bullish fractal AND price < 1w EMA50 (downtrend) AND volume > 2x average
            elif close[i] < bull_fractal and close[i] < ema_1w and vol > 2.0 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below bullish fractal OR trend reverses (price < 1w EMA50)
            if close[i] < bull_fractal or close[i] < ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above bearish fractal OR trend reverses (price > 1w EMA50)
            if close[i] > bear_fractal or close[i] > ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals