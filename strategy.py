#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Daily ATR Expansion with Volume and Trend Filter
# Hypothesis: ATR expansion (volatility increase) combined with price breakout and volume confirmation
# captures momentum moves in both bull and bear markets. Uses daily ATR to filter regime.
# Trend filter (price vs 50 EMA) ensures alignment with intermediate trend.
# Target: 15-30 trades/year (60-120 over 4 years).

name = "6h_daily_atr_expansion_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate daily ATR(14)
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # True Range
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - np.roll(daily_close, 1))
    tr3 = np.abs(daily_low - np.roll(daily_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR calculation with smoothing
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # ATR expansion: current ATR > 1.5x ATR from 5 periods ago
    atr_expansion = np.zeros_like(atr)
    for i in range(5, len(atr)):
        if atr[i-5] > 0:
            atr_expansion[i] = atr[i] > (1.5 * atr[i-5])
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    atr_expansion = np.roll(atr_expansion, 1)
    if len(atr_expansion) > 1:
        atr_expansion[0] = atr_expansion[1]
    else:
        atr_expansion[0] = False
    
    # Align to 6h timeframe
    atr_expansion_aligned = align_htf_to_ltf(prices, df_daily, atr_expansion)
    
    # Trend filter: price vs 50 EMA on 6h
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(ema_50[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: trend reversal or volatility contraction
            if close[i] < ema_50[i] or not atr_expansion_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: trend reversal or volatility contraction
            if close[i] > ema_50[i] or not atr_expansion_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: price above EMA with ATR expansion and volume
            if close[i] > ema_50[i] and atr_expansion_aligned[i] and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price below EMA with ATR expansion and volume
            elif close[i] < ema_50[i] and atr_expansion_aligned[i] and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals