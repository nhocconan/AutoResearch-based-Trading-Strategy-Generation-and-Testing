#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w trend filter and volume confirmation
# Long when Jaw < Teeth < Lips (bullish alignment) AND close > 1w EMA50 (uptrend) AND volume > 1.5 * 20-bar avg volume
# Short when Jaw > Teeth > Lips (bearish alignment) AND close < 1w EMA50 (downtrend) AND volume > 1.5 * 20-bar avg volume
# Exit when Alligator lines cross (alignment breaks)
# Uses discrete sizing 0.25 to control fee drag and drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Williams Alligator identifies trending vs ranging markets via smoothed SMAs
# 1w EMA50 ensures higher-timeframe trend alignment; volume confirms institutional participation

name = "12h_WilliamsAlligator_1wEMA50_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator lines (using 12h data)
    close_series = pd.Series(close)
    # Jaw: 13-period SMMA, 8 bars ahead
    jaw = close_series.rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMMA, 5 bars ahead
    teeth = close_series.rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMMA, 3 bars ahead
    lips = close_series.rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 12h timeframe (wait for completed HTF bar)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate volume confirmation: volume > 1.5 * 20-bar average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Alligator signals with trend and volume filters
            # Long: Jaw < Teeth < Lips (bullish alignment) AND uptrend AND volume spike
            if jaw[i] < teeth[i] and teeth[i] < lips[i] and close[i] > ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Jaw > Teeth > Lips (bearish alignment) AND downtrend AND volume spike
            elif jaw[i] > teeth[i] and teeth[i] > lips[i] and close[i] < ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks (jaw >= teeth or teeth >= lips)
            if jaw[i] >= teeth[i] or teeth[i] >= lips[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks (jaw <= teeth or teeth <= lips)
            if jaw[i] <= teeth[i] or teeth[i] <= lips[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals