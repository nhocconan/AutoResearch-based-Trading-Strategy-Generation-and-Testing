#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1d_cci_ema_v1
# Uses daily CCI (20) to identify overbought/oversold conditions and 6h EMA20 for trend direction.
# Long when daily CCI < -100 (oversold) and price > EMA20 on 6h (bullish momentum).
# Short when daily CCI > +100 (overbought) and price < EMA20 on 6h (bearish momentum).
# Designed for low trade frequency (target: 15-40 trades/year) to minimize fee drag.
# Works in both bull and bear markets by fading extremes in the direction of the trend.

name = "6h_1d_cci_ema_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for CCI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate CCI(20) on daily data
    typical_price = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    sma_tp = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    # Avoid division by zero
    cci = np.where(mad != 0, (typical_price - sma_tp) / (0.015 * mad), 0.0)
    
    # Align CCI to 6h timeframe (daily values update after daily bar closes)
    cci_aligned = align_htf_to_ltf(prices, df_1d, cci)
    
    # Calculate EMA20 on 6h for trend direction
    ema20_6h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(cci_aligned[i]) or np.isnan(ema20_6h[i]):
            signals[i] = 0.0
            continue
        
        # Long signal: daily CCI oversold (< -100) AND price above 6h EMA20
        if cci_aligned[i] < -100 and close[i] > ema20_6h[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: daily CCI overbought (> +100) AND price below 6h EMA20
        elif cci_aligned[i] > 100 and close[i] < ema20_6h[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: CCI returns to neutral zone (-100 to +100)
        elif -100 <= cci_aligned[i] <= 100 and position != 0:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals