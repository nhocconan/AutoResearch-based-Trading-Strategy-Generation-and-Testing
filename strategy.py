#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h pivot-based range breakout with volume confirmation and momentum filter
# - Uses 12-hour pivot points (PP, R1, S1, R2, S2) as dynamic support/resistance
# - Long when price breaks above R2 with volume > 1.5x 30-period average and positive momentum (close > open)
# - Short when price breaks below S2 with volume > 1.5x 30-period average and negative momentum (close < open)
# - Pivot levels provide structure in both trending and ranging markets
# - Volume filter ensures breakouts have conviction
# - Momentum filter reduces false breakouts
# - Position size 0.25 to balance risk and returns
# - Target: 60-120 trades over 4 years (15-30/year) to minimize fee drag
# - Works in bull markets (breakout continuation) and bear markets (mean reversion at S2/R2)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Load 12h data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h pivot points and support/resistance levels
    pivot = (high_12h + low_12h + close_12h) / 3.0
    r1 = 2 * pivot - low_12h
    s1 = 2 * pivot - high_12h
    r2 = pivot + (high_12h - low_12h)
    s2 = pivot - (high_12h - low_12h)
    
    # Align pivot levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_12h, pivot)
    r1_6h = align_htf_to_ltf(prices, df_12h, r1)
    s1_6h = align_htf_to_ltf(prices, df_12h, s1)
    r2_6h = align_htf_to_ltf(prices, df_12h, r2)
    s2_6h = align_htf_to_ltf(prices, df_12h, s2)
    
    # Volume filter: 30-period average (1 day of 6h bars)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(30, n):
        # Skip if any critical data is NaN
        if np.isnan(vol_ma[i]):
            continue
        
        if position == 0:
            # Long: Break above R2 with volume and momentum confirmation
            if (close[i] > r2_6h[i] and 
                volume[i] > vol_ma[i] * 1.5 and
                close[i] > open_price[i]):
                position = 1
                signals[i] = position_size
            # Short: Break below S2 with volume and momentum confirmation
            elif (close[i] < s2_6h[i] and 
                  volume[i] > vol_ma[i] * 1.5 and
                  close[i] < open_price[i]):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Break below S1 (mean reversion or trend exhaustion)
            if close[i] < s1_6h[i]:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Break above R1 (mean reversion or trend exhaustion)
            if close[i] > r1_6h[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_12h_Pivot_R2S2_Breakout_Volume_Momentum"
timeframe = "6h"
leverage = 1.0