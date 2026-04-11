#!/usr/bin/env python3
# 1h_4h_1d_volatility_breakout_v1
# Strategy: 1h volatility breakout with 4h trend filter and 1d volatility regime filter
# Timeframe: 1h
# Leverage: 1.0
# Hypothesis: Volatility expansion after low volatility periods captures breakouts in both bull and bear markets.
# Uses 4h for trend direction (avoid counter-trend trades) and 1d for volatility regime (only trade in high vol regimes).
# Designed for low trade frequency (~15-30/year) to avoid fee drag in challenging 1h timeframe.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_volatility_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (avoid Asian session noise)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA20 for trend filter
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Load 1d data ONCE before loop for volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d ATR(10) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr_10_1d = pd.Series(tr_1d).rolling(window=10, min_periods=10).mean().values
    atr_10_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_10_1d)
    
    # 1h ATR(10) for breakout threshold
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    tr = np.concatenate([[np.nan], tr])
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # 1h Donchian channel breakout (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid or outside session
        if (not in_session[i] or np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(atr_10_1d_aligned[i]) or np.isnan(atr_10[i]) or
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Volatility regime filter: only trade when 1d ATR is above its 30-period average
        if i >= 30:
            atr_ma_30 = np.nanmean(atr_10_1d_aligned[i-30:i]) if not np.isnan(np.nanmean(atr_10_1d_aligned[i-30:i])) else 0
            vol_regime = atr_10_1d_aligned[i] > 1.2 * atr_ma_30
        else:
            vol_regime = False
        
        # Trend filter: 4h EMA20 direction
        trend_bullish = close[i] > ema_20_4h_aligned[i]
        trend_bearish = close[i] < ema_20_4h_aligned[i]
        
        # Breakout signals
        breakout_up = high[i] > highest_20[i-1]
        breakdown_down = low[i] < lowest_20[i-1]
        
        # Entry conditions
        # Long: Breakout up + bullish trend + high volatility regime
        if breakout_up and trend_bullish and vol_regime and position != 1:
            position = 1
            signals[i] = 0.20
        # Short: Breakdown down + bearish trend + high volatility regime
        elif breakdown_down and trend_bearish and vol_regime and position != -1:
            position = -1
            signals[i] = -0.20
        # Exit: Opposite breakout or volatility collapse
        elif position == 1 and (low[i] < lowest_20[i-1] or not vol_regime):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (high[i] > highest_20[i-1] or not vol_regime):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals