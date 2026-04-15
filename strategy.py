#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1-day Trend Filter
# Williams %R(14) identifies overbought/oversold conditions: < -80 = oversold, > -20 = overbought.
# Combined with 1-day EMA50 trend filter: only take long signals when price above daily EMA50,
# and short signals when price below daily EMA50. This avoids counter-trend trades in strong trends.
# Uses volume confirmation (>1.5x 20-bar median volume) to ensure institutional participation.
# Designed to work in both bull and bear markets by following the higher timeframe trend.
# Discrete sizing (0.25) limits trade frequency to ~25-35 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Williams %R calculation (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(14, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_threshold[i])):
            continue
        
        # Long: Williams %R < -80 (oversold), price above 1d EMA50, volume spike
        if (williams_r[i] < -80 and close[i] > ema_1d_aligned[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: Williams %R > -20 (overbought), price below 1d EMA50, volume spike
        elif (williams_r[i] > -20 and close[i] < ema_1d_aligned[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: Williams %R returns to neutral range (-50) or trend fails
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (williams_r[i] >= -50 or close[i] <= ema_1d_aligned[i])) or
               (signals[i-1] == -0.25 and (williams_r[i] <= -50 or close[i] >= ema_1d_aligned[i])))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_WilliamsR_1dEMA_Volume"
timeframe = "4h"
leverage = 1.0