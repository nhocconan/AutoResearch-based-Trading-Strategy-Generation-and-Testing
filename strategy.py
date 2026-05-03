#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA(50) trend filter and ATR-based volatility filter
# Designed to capture strong trending moves aligned with daily momentum while filtering low-volatility chop.
# Uses discrete position sizing (0.30) to balance profit potential and drawdown control.
# Works in bull/bear markets by following 1d EMA50 direction and requiring ATR-based volatility confirmation.
# Target: 75-200 trades over 4 years (19-50/year) to minimize fee drag while maintaining edge.

name = "4h_Donchian20_1dEMA50_ATR_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for volatility filter on 4h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start from 60 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: ATR > 0.5 * 20-period ATR mean (avoid low-vol chop)
        atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
        volatility_filter = atr[i] > (0.5 * atr_ma[i]) if not np.isnan(atr_ma[i]) else False
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper + above 1d EMA50 + volatility filter
            if close[i] > highest_high[i] and price_above_ema and volatility_filter:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Donchian lower + below 1d EMA50 + volatility filter
            elif close[i] < lowest_low[i] and price_below_ema and volatility_filter:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian lower or loses 1d trend alignment
            if close[i] < lowest_low[i] or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price breaks above Donchian upper or loses 1d trend alignment
            if close[i] > highest_high[i] or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals