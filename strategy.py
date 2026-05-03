#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR-based volatility filter
# Donchian channels provide robust trend-following structure across market regimes
# 1d EMA50 ensures alignment with higher timeframe trend to avoid counter-trend trades
# ATR filter ensures trades only occur in sufficient volatility environments
# Target: 100-180 total trades over 4 years (25-45/year) to balance edge and fee drag

name = "4h_Donchian20_1dEMA50_ATRFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) from previous bar (no look-ahead)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate ATR(14) for volatility filter
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volatility filter: ATR > 0.5 * 20-period EMA of ATR (ensures sufficient volatility)
    atr_ema_20 = pd.Series(atr).ewm(span=20, adjust=False, min_periods=20).mean().values
    volatility_filter = atr > (0.5 * atr_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volatility_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout signals with 1d trend filter and volatility filter
        # Long: Break above upper Donchian + price above 1d EMA50 + volatility filter
        # Short: Break below lower Donchian + price below 1d EMA50 + volatility filter
        if position == 0:
            if close[i] > highest_high[i] and close[i] > ema_50_1d_aligned[i] and volatility_filter[i]:
                signals[i] = 0.25
                position = 1
            elif close[i] < lowest_low[i] and close[i] < ema_50_1d_aligned[i] and volatility_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below lower Donchian (trend reversal) OR below 1d EMA50
            if close[i] < lowest_low[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above upper Donchian (trend reversal) OR above 1d EMA50
            if close[i] > highest_high[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals