#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with daily trend filter and volume confirmation
# Long when price breaks above bullish fractal AND price above daily EMA(50) AND volume confirmation
# Short when price breaks below bearish fractal AND price below daily EMA(50) AND volume confirmation
# Exit when price crosses back through daily EMA(50)
# Williams Fractals require 2-bar confirmation (center bar + 2 bars after)
# Williams Fractals provide high-probability reversal points with built-in look-ahead protection
# Daily EMA(50) filters for intermediate-term trend alignment
# Volume confirmation ensures breakouts have participation
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA(50)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams Fractals on daily data
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-3] < high[n-2] and high[n+1] < high[n]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-3] > low[n-2] and low[n+1] > low[n]
    n_1d = len(high_1d)
    bearish_fractal = np.full(n_1d, np.nan)
    bullish_fractal = np.full(n_1d, np.nan)
    
    for i in range(2, n_1d - 2):
        # Bearish fractal (peak)
        if (high[i-2] < high[i-1] and 
            high[i] > high[i-1] and 
            high[i-3] < high[i-2] and 
            high[i+1] < high[i]):
            bearish_fractal[i] = high[i]
        
        # Bullish fractal (trough)
        if (low[i-2] > low[i-1] and 
            low[i] < low[i-1] and 
            low[i-3] > low[i-2] and 
            low[i+1] > low[i]):
            bullish_fractal[i] = low[i]
    
    # Williams Fractals need 2 extra bars for confirmation after the center bar
    bearish_fractal_confirmed = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_confirmed = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Align daily EMA to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 20)  # Need EMA and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(bearish_fractal_confirmed[i]) or 
            np.isnan(bullish_fractal_confirmed[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above/below daily EMA(50)
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Look for fractal breakouts
            # Long: price breaks above bullish fractal AND price above daily EMA AND volume confirmation
            if (not np.isnan(bullish_fractal_confirmed[i]) and 
                close[i] > bullish_fractal_confirmed[i] and 
                price_above_ema and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below bearish fractal AND price below daily EMA AND volume confirmation
            elif (not np.isnan(bearish_fractal_confirmed[i]) and 
                  close[i] < bearish_fractal_confirmed[i] and 
                  price_below_ema and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below daily EMA(50)
            if close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above daily EMA(50)
            if close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_WilliamsFractal_DailyEMA_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0