#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1d EMA trend filter and volume confirmation.
# Uses fractal breaks for directional entries, 1d EMA for trend filter, volume spike for confirmation.
# Designed to work in bull (breakouts with trend) and bear (mean reversion via fractal rejection).
# Target: 50-150 total trades over 4 years (~12-37/year) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams Fractals and EMA
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Fractals (bearish: high[2] is highest of 5, bullish: low[2] is lowest of 5)
    n1 = len(high_1d)
    bearish_fractal = np.full(n1, np.nan)
    bullish_fractal = np.full(n1, np.nan)
    
    for i in range(2, n1 - 2):
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]  # Bearish fractal at i (sell signal)
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]   # Bullish fractal at i (buy signal)
    
    # Williams Fractals need 2 extra bars for confirmation (the two bars after the center)
    bearish_fractal_confirmed = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_confirmed = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate daily EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_6h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 1.8 * 20-period average (balanced to avoid overtrading)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 34  # Need daily EMA34 and volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bearish_fractal_confirmed[i]) or 
            np.isnan(bullish_fractal_confirmed[i]) or 
            np.isnan(ema34_6h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 1.8x average (moderate to balance signals)
        volume_filter = volume[i] > (1.8 * volume_ma20[i])
        
        # Trend filter: price above/below daily EMA34
        price_above_ema = close[i] > ema34_6h[i]
        price_below_ema = close[i] < ema34_6h[i]
        
        # Price relative to Williams Fractal levels
        price_above_bearish = close[i] > bearish_fractal_confirmed[i]
        price_below_bullish = close[i] < bullish_fractal_confirmed[i]
        
        if position == 0:
            # Long: Price breaks above bearish fractal with volume and above daily EMA34
            if (price_above_bearish and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below bullish fractal with volume and below daily EMA34
            elif (price_below_bullish and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below bullish fractal OR below daily EMA34
            if (close[i] < bullish_fractal_confirmed[i]) or (close[i] < ema34_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above bearish fractal OR above daily EMA34
            if (close[i] > bearish_fractal_confirmed[i]) or (close[i] > ema34_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsFractal_Breakout_EMA34_Volume"
timeframe = "6h"
leverage = 1.0