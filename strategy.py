#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal + 1d trend filter (EMA34) + volume confirmation.
# Williams Fractal identifies reversal points; EMA34 defines trend direction.
# Volume confirms breakout strength. Works in both bull/bear by trading with trend.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within profitable range.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate Williams Fractals on daily (requires 5-bar window)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Bullish fractal: low[n-2] < low[n-1] and low[n] < low[n-1] and low[n+1] > low[n-1] and low[n+2] > low[n-1]
    # Bearish fractal: high[n-2] > high[n-1] and high[n] > high[n-1] and high[n+1] < high[n-1] and high[n+2] < high[n-1]
    bullish_fractal = np.zeros(len(high_1d), dtype=bool)
    bearish_fractal = np.zeros(len(high_1d), dtype=bool)
    
    for i in range(2, len(high_1d) - 2):
        if (low_1d[i-2] > low_1d[i-1] and low_1d[i] > low_1d[i-1] and 
            low_1d[i+1] < low_1d[i-1] and low_1d[i+2] < low_1d[i-1]):
            bearish_fractal[i-1] = True  # Bearish fractal at i-1
        if (high_1d[i-2] < high_1d[i-1] and high_1d[i] < high_1d[i-1] and 
            high_1d[i+1] > high_1d[i-1] and high_1d[i+2] > high_1d[i-1]):
            bullish_fractal[i-1] = True  # Bullish fractal at i-1
    
    # Convert to float arrays for alignment (1.0 where fractal exists, 0.0 otherwise)
    bullish_fractal_float = bullish_fractal.astype(float)
    bearish_fractal_float = bearish_fractal.astype(float)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate daily volume and its 20-period average for volume filter
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all data to 6-hour timeframe
    # Williams fractals need 2 extra bars for confirmation (as per rule 2b)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal_float, additional_delay_bars=2)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal_float, additional_delay_bars=2)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(bullish_fractal_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 6h volume > 1.3x daily volume MA (adjusted for 6h)
        # 4 6h periods per day, so daily MA/4 = approximate 6h period MA
        volume_6h_approx_ma = volume_ma_20_1d_aligned[i] / 4
        volume_condition = volume[i] > (volume_6h_approx_ma * 1.3)
        
        # Fractal signals (using previous bar to avoid look-ahead)
        bullish_fractal_signal = bullish_fractal_aligned[i-1] > 0.5
        bearish_fractal_signal = bearish_fractal_aligned[i-1] > 0.5
        
        # Trend filter: price above/below EMA34
        price_above_ema = close[i] > ema_34_aligned[i]
        price_below_ema = close[i] < ema_34_aligned[i]
        
        # Entry conditions
        if position == 0:
            # Long: bullish fractal + price above EMA34 + volume
            if bullish_fractal_signal and price_above_ema and volume_condition:
                position = 1
                signals[i] = position_size
            # Short: bearish fractal + price below EMA34 + volume
            elif bearish_fractal_signal and price_below_ema and volume_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: bearish fractal or price below EMA34
            if bearish_fractal_signal or price_below_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: bullish fractal or price above EMA34
            if bullish_fractal_signal or price_above_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_Williams_Fractal_EMA34_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0