#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Williams Fractal breakout with daily EMA34 filter and volume confirmation.
# Long when: Price breaks above bullish fractal high, daily EMA34 upward, volume > 1.5x 30-period average
# Short when: Price breaks below bearish fractal low, daily EMA34 downward, volume > 1.5x 30-period average
# Exit when: Price crosses back through the fractal level in opposite direction
# Williams Fractals identify key swing points, EMA34 filters trend, volume confirms breakout strength.
# Target: 15-30 trades/year per symbol. Works in bull (buy breakouts) and bear (sell breakdowns).
name = "12h_WilliamsFractal_Breakout_EMA34_Volume"
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
    
    # 1-day data for Williams Fractals and EMA34
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Fractals: bearish (high) and bullish (low) fractals
    # Bearish fractal: middle bar has highest high, 2 lower highs on each side
    # Bullish fractal: middle bar has lowest low, 2 higher lows on each side
    n1d = len(high_1d)
    bearish_fractal = np.full(n1d, np.nan)
    bullish_fractal = np.full(n1d, np.nan)
    
    for i in range(2, n1d - 2):
        # Bearish fractal: current high is highest of 5 bars
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        # Bullish fractal: current low is lowest of 5 bars
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Calculate EMA34 on daily data for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1D data to 12H timeframe
    # Williams fractals need 2 extra bars for confirmation (pattern completes after 2 bars)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 30-period volume average for confirmation
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA34 calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        bearish_fractal_level = bearish_fractal_aligned[i]
        bullish_fractal_level = bullish_fractal_aligned[i]
        ema34 = ema34_1d_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_30[i]
        
        if position == 0:
            # Long entry: Price breaks above bullish fractal, EMA34 upward, volume spike
            if (price > bullish_fractal_level and close[i-1] <= bullish_fractal_level and 
                ema34 > ema34_1d_aligned[i-1] and vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below bearish fractal, EMA34 downward, volume spike
            elif (price < bearish_fractal_level and close[i-1] >= bearish_fractal_level and 
                  ema34 < ema34_1d_aligned[i-1] and vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses back below bullish fractal level
            if price < bullish_fractal_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses back above bearish fractal level
            if price > bearish_fractal_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals