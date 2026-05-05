#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1d trend filter and volume confirmation
# Long when price breaks above latest bearish fractal (swing high) AND 1d close > 1d EMA50 AND volume > 1.5x 20-period average
# Short when price breaks below latest bullish fractal (swing low) AND 1d close < 1d EMA50 AND volume > 1.5x 20-period average
# Exit when price crosses 1d EMA50 (trend reversal)
# Williams Fractals provide swing points that work in both trending and ranging markets
# Fractal breakouts capture momentum after consolidation, reduced fakeouts with volume/trend filters
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe

name = "6h_WilliamsFractal_Breakout_1dEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Fractals on 1d data
    # Bearish fractal: high[n-2] is highest of [high[n-4], high[n-3], high[n-2], high[n-1], high[n]]
    # Bullish fractal: low[n-2] is lowest of [low[n-4], low[n-3], low[n-2], low[n-1], low[n]]
    bearish_fractal = np.full(len(df_1d), np.nan)
    bullish_fractal = np.full(len(df_1d), np.nan)
    
    for i in range(2, len(df_1d) - 2):
        # Bearish fractal: center high is highest of 5 bars
        if (df_1d['high'].iloc[i] >= df_1d['high'].iloc[i-2] and 
            df_1d['high'].iloc[i] >= df_1d['high'].iloc[i-1] and
            df_1d['high'].iloc[i] >= df_1d['high'].iloc[i+1] and
            df_1d['high'].iloc[i] >= df_1d['high'].iloc[i+2]):
            bearish_fractal[i] = df_1d['high'].iloc[i]
        
        # Bullish fractal: center low is lowest of 5 bars
        if (df_1d['low'].iloc[i] <= df_1d['low'].iloc[i-2] and 
            df_1d['low'].iloc[i] <= df_1d['low'].iloc[i-1] and
            df_1d['low'].iloc[i] <= df_1d['low'].iloc[i+1] and
            df_1d['low'].iloc[i] <= df_1d['low'].iloc[i+2]):
            bullish_fractal[i] = df_1d['low'].iloc[i]
    
    # Forward fill fractal levels to get most recent swing point
    bearish_fractal = pd.Series(bearish_fractal).ffill().values
    bullish_fractal = pd.Series(bullish_fractal).ffill().values
    
    # Align fractal levels to 6h timeframe with extra delay (fractals need 2-bar confirmation)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above latest bearish fractal (swing high) AND 1d close > 1d EMA50 AND volume spike
            if (close[i] > bearish_fractal_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below latest bullish fractal (swing low) AND 1d close < 1d EMA50 AND volume spike
            elif (close[i] < bullish_fractal_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1d EMA50 (trend reversal)
            if close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1d EMA50 (trend reversal)
            if close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals