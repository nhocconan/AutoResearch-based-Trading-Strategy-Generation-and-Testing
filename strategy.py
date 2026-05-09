#!/usr/bin/env python3
# Hypothesis: 6h timeframe with daily ATR-based volatility regime filter and daily EMA34 trend filter.
# Uses daily ATR(14) > daily SMA(ATR14, 50) to identify high volatility regimes for breakout entries.
# Daily EMA34 provides trend direction to avoid counter-trend trades.
# Volatility regime filter helps capture momentum bursts while avoiding choppy periods.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "6h_ATR_Volatility_Regime_EMA34_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR and EMA calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily SMA of ATR(50) for volatility regime
    atr_sma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    
    # Volatility regime: ATR > SMA(ATR) indicates high volatility regime
    vol_regime = atr_14 > atr_sma_50
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily indicators to 6h timeframe
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Trend conditions
    trend_up = close > ema_34_aligned
    trend_down = close < ema_34_aligned
    
    # Breakout conditions: price breaks above/below 6-period high/low
    high_6 = pd.Series(high).rolling(window=6, min_periods=6).max().values
    low_6 = pd.Series(low).rolling(window=6, min_periods=6).min().values
    
    breakout_up = close > high_6
    breakout_down = close < low_6
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_regime_aligned[i]) or np.isnan(ema_34_aligned[i]) or
            np.isnan(breakout_up[i]) or np.isnan(breakout_down[i]) or
            np.isnan(trend_up[i]) or np.isnan(trend_down[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout up + high volatility regime + uptrend
            if breakout_up[i] and vol_regime_aligned[i] and trend_up[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout down + high volatility regime + downtrend
            elif breakout_down[i] and vol_regime_aligned[i] and trend_down[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: volatility regime ends or trend reverses
            if not vol_regime_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: volatility regime ends or trend reverses
            if not vol_regime_aligned[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals