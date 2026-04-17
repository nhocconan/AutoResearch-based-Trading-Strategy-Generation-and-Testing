#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (SMA50)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma50_1d = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    # Get daily data for daily range (for position sizing)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    daily_range = high_1d - low_1d
    
    # Daily ATR (14-period) for volatility filter
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(np.abs(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])  # Align with indices
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need weekly SMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(sma50_1d[i]) or np.isnan(atr14[i]):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR > 50-day average ATR
        atr_ma50 = pd.Series(atr14).rolling(window=50, min_periods=50).mean().values
        if np.isnan(atr_ma50[i]) or atr14[i] <= atr_ma50[i]:
            signals[i] = 0.0
            continue
        
        # Position sizing based on daily volatility (inverse volatility scaling)
        # Base size 0.25, scaled by ATR
        atr_norm = atr14[i] / atr_ma50[i] if atr_ma50[i] > 0 else 1.0
        size = 0.25 / atr_norm  # Inverse volatility: smaller size when vol high
        size = min(max(size, 0.10), 0.40)  # Clamp between 0.10 and 0.40
        
        if position == 0:
            # Long: price above weekly SMA50
            if close[i] > sma50_1d[i]:
                signals[i] = size
                position = 1
            # Short: price below weekly SMA50
            elif close[i] < sma50_1d[i]:
                signals[i] = -size
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below weekly SMA50
            if close[i] < sma50_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        
        elif position == -1:
            # Exit short: price crosses above weekly SMA50
            if close[i] > sma50_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklySMA50_Trend_InverseVol"
timeframe = "1d"
leverage = 1.0