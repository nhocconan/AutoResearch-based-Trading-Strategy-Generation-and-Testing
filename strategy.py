#!/usr/bin/env python3
# 1d_1w_camarilla_pivot_volume_v1
# Strategy: 1d Camarilla pivot breakout with volume confirmation and 1w trend filter
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels provide high-probability support/resistance levels.
# In trending markets (1w EMA50), price breaking above/below pivot levels with volume
# continuation signals strong momentum. In ranging markets, pivot reversals offer mean reversion.
# Designed for low frequency (<25/year) to minimize fee drag while capturing significant moves.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_pivot_volume_v1"
timeframe = "1d"
leverage = 1.0

def calculate_camarilla_pivots(high, low, close):
    """Calculate Camarilla pivot levels for the day."""
    typical_price = (high + low + close) / 3
    range_val = high - low
    if range_val <= 0:
        return close, close, close, close, close, close, close, close
    # Camarilla levels
    S1 = close - (range_val * 1.1 / 12)
    S2 = close - (range_val * 1.1 / 6)
    S3 = close - (range_val * 1.1 / 4)
    S4 = close - (range_val * 1.1 / 2)
    R1 = close + (range_val * 1.1 / 12)
    R2 = close + (range_val * 1.1 / 6)
    R3 = close + (range_val * 1.1 / 4)
    R4 = close + (range_val * 1.1 / 2)
    return S1, S2, S3, S4, R1, R2, R3, R4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-calculate Camarilla pivots for each day (using previous day's data)
    camarilla_S1 = np.full(n, np.nan)
    camarilla_S2 = np.full(n, np.nan)
    camarilla_S3 = np.full(n, np.nan)
    camarilla_S4 = np.full(n, np.nan)
    camarilla_R1 = np.full(n, np.nan)
    camarilla_R2 = np.full(n, np.nan)
    camarilla_R3 = np.full(n, np.nan)
    camarilla_R4 = np.full(n, np.nan)
    
    for i in range(1, n):
        S1, S2, S3, S4, R1, R2, R3, R4 = calculate_camarilla_pivots(
            high[i-1], low[i-1], close[i-1]
        )
        camarilla_S1[i] = S1
        camarilla_S2[i] = S2
        camarilla_S3[i] = S3
        camarilla_S4[i] = S4
        camarilla_R1[i] = R1
        camarilla_R2[i] = R2
        camarilla_R3[i] = R3
        camarilla_R4[i] = R4
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_R1[i]) or np.isnan(camarilla_S1[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Entry logic: Camarilla breakout + volume + trend alignment
        if (close[i] > camarilla_R1[i] and vol_confirm[i] and uptrend and position != 1):
            position = 1
            signals[i] = 0.25
        elif (close[i] < camarilla_S1[i] and vol_confirm[i] and downtrend and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: price returns to midpoint or trend change
        elif position == 1 and (close[i] < (camarilla_R1[i] + camarilla_S1[i]) / 2 or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > (camarilla_R1[i] + camarilla_S1[i]) / 2 or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals