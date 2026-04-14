#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with daily trend filter and volume confirmation
# Williams Fractals identify swing highs/lows that act as support/resistance.
# Breakouts above/below recent fractals with volume and daily EMA trend filter
# capture institutional flow while avoiding false breakouts in chop.
# Works in bull/bear by using daily EMA trend filter (long only above EMA, short only below EMA)
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams Fractals and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA(50) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Fractals (5-bar pattern: high/low surrounded by 2 lower highs/lows)
    n1 = len(high_1d)
    bullish_fractal = np.zeros(n1, dtype=bool)
    bearish_fractal = np.zeros(n1, dtype=bool)
    
    for i in range(2, n1 - 2):
        # Bullish fractal: low[i] is lowest among low[i-2:i+3]
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = True
        # Bearish fractal: high[i] is highest among high[i-2:i+3]
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = True
    
    # For breakout signals, we need the most recent fractal levels
    # Convert to float arrays with NaN where no fractal
    bullish_fractal_level = np.where(bullish_fractal, low_1d, np.nan)
    bearish_fractal_level = np.where(bearish_fractal, high_1d, np.nan)
    
    # Forward fill to get the most recent fractal level
    bullish_series = pd.Series(bullish_fractal_level)
    bearish_series = pd.Series(bearish_fractal_level)
    recent_bullish = bullish_series.ffill().values
    recent_bearish = bearish_series.ffill().values
    
    # Align fractal levels to 6h timeframe
    # Williams fractals need 2 extra bars for confirmation (pattern completes at bar i, confirmed at i+2)
    bullish_aligned = align_htf_to_ltf(prices, df_1d, recent_bullish, additional_delay_bars=2)
    bearish_aligned = align_htf_to_ltf(prices, df_1d, recent_bearish, additional_delay_bars=2)
    
    # Volume confirmation: volume > 1.5x average volume (24-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=24, min_periods=24).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 24  # for volume average
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price breaks above most recent bearish fractal (resistance) with volume filter AND above daily EMA50
            if (price > bearish_aligned[i] and price > ema_50_1d_aligned[i] and 
                vol > 1.5 * avg_vol[i]):
                position = 1
                signals[i] = position_size
            # Short: price breaks below most recent bullish fractal (support) with volume filter AND below daily EMA50
            elif (price < bullish_aligned[i] and price < ema_50_1d_aligned[i] and 
                  vol > 1.5 * avg_vol[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below most recent bullish fractal (support) OR below daily EMA50
            if price < bullish_aligned[i] or price < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above most recent bearish fractal (resistance) OR above daily EMA50
            if price > bearish_aligned[i] or price > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Williams_Fractal_Breakout_EMA_Volume"
timeframe = "6h"
leverage = 1.0