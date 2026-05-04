#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 12h trend filter (EMA50) + volume confirmation (>1.5x 20 EMA)
# In trending markets (12h close > EMA50), trade breakouts in trend direction: long on upper band breakout, short on lower band breakdown.
# Volume confirmation reduces false breakouts. Designed for 6h timeframe targeting 75-150 total trades over 4 years.
# Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown.

name = "6h_Donchian20_12hEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 6h Donchian channels (20-period)
    # Upper band: highest high over last 20 periods
    high_series = pd.Series(high)
    upper_band = high_series.rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over last 20 periods
    low_series = pd.Series(low)
    lower_band = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Only trade in direction of 12h trend
            if close[i] > ema_50_12h_aligned[i]:  # Uptrend
                # Long on upper band breakout with volume confirmation
                if close[i] > upper_band[i] and volume_confirm:
                    signals[i] = 0.25
                    position = 1
            else:  # Downtrend
                # Short on lower band breakdown with volume confirmation
                if close[i] < lower_band[i] and volume_confirm:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price retouches lower band OR 12h trend turns down
            if close[i] < lower_band[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches upper band OR 12h trend turns up
            if close[i] > upper_band[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals