#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with daily EMA34 trend filter and volume confirmation.
# Uses daily Williams fractals (bearish/bullish) identified from previous completed day.
# Enters long when price breaks above most recent bearish fractal with volume and above daily EMA34.
# Enters short when price breaks below most recent bullish fractal with volume and below daily EMA34.
# Designed to capture institutional breakouts with low turnover (target: 12-37 trades/year).
# Works in bull markets (breakout momentum) and bear markets (mean reversion via fractal rejection).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams fractal calculation and EMA
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams fractals on daily data
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    n1d = len(high_1d)
    bearish_fractal = np.full(n1d, np.nan)
    bullish_fractal = np.full(n1d, np.nan)
    
    for i in range(2, n1d - 2):
        # Bearish fractal (peak)
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i] < high_1d[i-1] and
            high_1d[i-3] < high_1d[i-1] and
            high_1d[i+1] < high_1d[i-1]):
            bearish_fractal[i-1] = high_1d[i-1]
        
        # Bullish fractal (trough)
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i] > low_1d[i-1] and
            low_1d[i-3] > low_1d[i-1] and
            low_1d[i+1] > low_1d[i-1]):
            bullish_fractal[i-1] = low_1d[i-1]
    
    # Calculate daily EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 6h timeframe
    bearish_fractal_6h = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_6h = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    ema34_6h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 2.0 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need sufficient data for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bearish_fractal_6h[i]) or 
            np.isnan(bullish_fractal_6h[i]) or 
            np.isnan(ema34_6h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 2.0x average (strict to reduce trades)
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Trend filter: price above/below daily EMA34
        price_above_ema = close[i] > ema34_6h[i]
        price_below_ema = close[i] < ema34_6h[i]
        
        # Price relative to Williams fractals
        price_above_bearish = close[i] > bearish_fractal_6h[i]
        price_below_bullish = close[i] < bullish_fractal_6h[i]
        
        if position == 0:
            # Long: Price breaks above most recent bearish fractal with volume and above daily EMA34
            if (price_above_bearish and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below most recent bullish fractal with volume and below daily EMA34
            elif (price_below_bullish and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below daily EMA34
            if close[i] < ema34_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above daily EMA34
            if close[i] > ema34_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsFractal_Breakout_EMA34_Volume"
timeframe = "6h"
leverage = 1.0