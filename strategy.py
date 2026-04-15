#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volatility filter + volume confirmation
# Long when price breaks above Donchian(20) high + volatility contraction + volume spike
# Short when price breaks below Donchian(20) low + volatility contraction + volume spike
# Works in bull (breakouts continue) and bear (breakdowns continue)
# Uses discrete sizing (0.25) to limit overtrading and fee drag
# 1d volatility filter ensures we trade only during expansion phases (ATR ratio > 1.2)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d ATR for volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 4-period ATR for current volatility (use 4h data)
    tr_4h = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_4h[0] = high[0] - low[0]
    atr_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volatility ratio: current 4h ATR / 1d ATR
    vol_ratio = atr_4h / atr_1d_aligned
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: breakout above Donchian high + volatility expansion + volume spike
        if (close[i] > donch_high[i] and 
            vol_ratio[i] > 1.2 and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: breakdown below Donchian low + volatility expansion + volume spike
        elif (close[i] < donch_low[i] and 
              vol_ratio[i] > 1.2 and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: volatility contraction or mean reversion to mid-point
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (vol_ratio[i] < 1.0 or close[i] < (donch_high[i] + donch_low[i]) / 2)) or
               (signals[i-1] == -0.25 and (vol_ratio[i] < 1.0 or close[i] > (donch_high[i] + donch_low[i]) / 2)))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Donchian_Breakout_Vol_VolatilityFilter"
timeframe = "4h"
leverage = 1.0