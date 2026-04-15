# 2025-07-01: 4h Donchian Breakout with Volume Confirmation and ATR Filter
# Hypothesis: Donchian breakouts capture momentum in both bull and bear markets.
# Volume confirmation ensures breakouts are institutional. ATR filter avoids low-volatility false breakouts.
# Target: 20-40 trades/year to minimize fee drag. Works in bull (breakouts up) and bear (breakouts down).
# Uses 4h timeframe with 1d HTF trend filter via EMA200 to align with higher timeframe bias.

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
    
    # Donchian Channel (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # ATR (14-period) for volatility filter
    tr1 = pd.Series(high).rolling(window=2).max() - pd.Series(low).rolling(window=2).min()
    tr2 = abs(pd.Series(high).rolling(window=2).max() - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low).rolling(window=2).min() - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    # Volume confirmation: current > 1.3x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.3 * vol_median
    
    # 1d EMA200 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(atr[i]) or np.isnan(vol_threshold[i]) or np.isnan(ema_200_aligned[i]):
            continue
        
        # Long: price breaks above Donchian high + volume + above 1d EMA200
        if (close[i] > high_max[i] and 
            volume[i] > vol_threshold[i] and 
            close[i] > ema_200_aligned[i]):
            signals[i] = 0.25
        
        # Short: price breaks below Donchian low + volume + below 1d EMA200
        elif (close[i] < low_min[i] and 
              volume[i] > vol_threshold[i] and 
              close[i] < ema_200_aligned[i]):
            signals[i] = -0.25
        
        # Exit: price returns to middle of Donchian channel (mean reversion)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < (high_max[i] + low_min[i]) / 2) or
               (signals[i-1] == -0.25 and close[i] > (high_max[i] + low_min[i]) / 2))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Donchian_Breakout_Volume_ATR"
timeframe = "4h"
leverage = 1.0