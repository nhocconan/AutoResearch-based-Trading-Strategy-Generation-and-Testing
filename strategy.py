#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when price breaks above upper band AND close > 1d EMA34 (uptrend) AND volume > 2.0 * 20-bar avg volume
# Short when price breaks below lower band AND close < 1d EMA34 (downtrend) AND volume > 2.0 * 20-bar avg volume
# Exit when price retraces to the midpoint of the Donchian channel (mean reversion to equilibrium)
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# 1d EMA34 provides strong trend filter between 4h and 1w for better regime adaptation
# Volume spike threshold increased to 2.0x to reduce false breakouts and lower trade frequency
# Midpoint exit works in ranging markets and captures mean reversion after breakout failure

name = "4h_Donchian20_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channel for 4h timeframe (based on previous 20 bars)
    upper = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lower = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    midpoint = (upper + lower) / 2.0
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 4h timeframe (wait for completed HTF bar)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-bar average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Donchian breakout signals with trend and volume filters
            # Long: Break above upper band AND uptrend AND volume spike
            if close[i] > upper[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower band AND downtrend AND volume spike
            elif close[i] < lower[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price retraces to midpoint (mean reversion)
            if close[i] <= midpoint[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price retraces to midpoint (mean reversion)
            if close[i] >= midpoint[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals