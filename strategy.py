#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Donchian breakout (20-period) with 1h EMA50 trend filter and 1d volume spike confirmation
# Long: Price breaks above 20-day high (excluding current) + 1h close > 1h EMA50 + volume > 1.5x 20-day avg volume
# Short: Price breaks below 20-day low (excluding current) + 1h close < 1h EMA50 + volume > 1.5x 20-day avg volume
# Exit: Price crosses below/above 20-day SMA (for long/short respectively)
# Designed for 1d timeframe targeting 10-25 trades/year. Works in bull/bear markets via trend filter and volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for indicators
        return np.zeros(n)
    
    # Extract price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1h data for trend filter (EMA50 and close)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 50:
        return np.zeros(n)
    
    close_1h = df_1h['close'].values
    # Calculate 1h EMA50
    ema50_1h = pd.Series(close_1h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align 1h indicators to 1d timeframe
    close_1h_aligned = align_htf_to_ltf(prices, df_1h, close_1h)
    ema50_1h_aligned = align_htf_to_ltf(prices, df_1h, ema50_1h)
    
    # 1d Donchian bands (20-period, excluding current bar)
    upper_band = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lower_band = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # 1d 20-day SMA for exit
    sma_20d = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # 1d 20-day average volume (excluding current bar) for volume spike
    vol_avg_20d = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    # Initialize signals and position tracker
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(n):
        # Skip if any data is not ready
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(sma_20d[i]) or np.isnan(vol_avg_20d[i]) or
            np.isnan(close_1h_aligned[i]) or np.isnan(ema50_1h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout conditions
            if (close[i] > upper_band[i] and 
                close_1h_aligned[i] > ema50_1h_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20d[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout conditions
            elif (close[i] < lower_band[i] and 
                  close_1h_aligned[i] < ema50_1h_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20d[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price below 20-day SMA
            if close[i] < sma_20d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        else:  # position == -1
            # Short exit: price above 20-day SMA
            if close[i] > sma_20d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1hEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0