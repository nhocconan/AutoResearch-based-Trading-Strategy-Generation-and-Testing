#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ATR volatility filter and volume confirmation
# Uses discrete position sizing (0.25) to limit overtrading and fee drag
# Volatility filter ensures trades only in sufficient volatility regimes
# Works in bull (breakouts continue) and bear (mean reversion at bands)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day ATR for volatility filter (calculated once)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 4-hour Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_donchian = high_series.rolling(window=20, min_periods=20).max()
    lower_donchian = low_series.rolling(window=20, min_periods=20).min()
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    # ATR-based volatility filter: require ATR > 0.3 * median ATR
    atr_median = pd.Series(atr_1d_aligned).rolling(window=50, min_periods=1).median()
    vol_filter = atr_1d_aligned > 0.3 * atr_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or 
            np.isnan(vol_threshold[i]) or np.isnan(vol_filter[i])):
            continue
        
        # Long: close breaks above upper Donchian + volume + volatility filter
        if close[i] > upper_donchian[i] and volume[i] > vol_threshold[i] and vol_filter[i]:
            signals[i] = 0.25
        
        # Short: close breaks below lower Donchian + volume + volatility filter
        elif close[i] < lower_donchian[i] and volume[i] > vol_threshold[i] and vol_filter[i]:
            signals[i] = -0.25
        
        # Exit: close crosses back inside bands (mean reversion)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < upper_donchian[i]) or
               (signals[i-1] == -0.25 and close[i] > lower_donchian[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Donchian_Breakout_Volume_VolFilter"
timeframe = "4h"
leverage = 1.0