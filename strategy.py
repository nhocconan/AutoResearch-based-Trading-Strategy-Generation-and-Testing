#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Trend Pullback with 4h Trend Filter and Volume Spike
# Uses 4h EMA20 for trend direction, enters on 1h pullback to EMA20 with volume confirmation.
# Long when 4h EMA20 rising, 1h price touches EMA20 from below with volume > 1.5x median.
# Short when 4h EMA20 falling, 1h price touches EMA20 from above with volume > 1.5x median.
# Uses 1h EMA20 as dynamic support/resistance. Designed to capture trend continuations
# in both bull and bear markets by following higher timeframe trend.
# Conservative sizing (0.20) to limit trade frequency and avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h EMA20 for trend direction
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    ema_4h_slope = np.diff(ema_4h_aligned, prepend=ema_4h_aligned[0])
    
    # 1h EMA20 for entry level
    ema_1h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_4h_slope[i]) or np.isnan(ema_1h[i]) or 
            np.isnan(vol_threshold[i])):
            continue
        
        # Long: 4h EMA20 rising, 1h low touches EMA1H from below, volume spike
        if (ema_4h_slope[i] > 0 and 
            low[i] <= ema_1h[i] and 
            high[i] > ema_1h[i] and  # crossed above
            volume[i] > vol_threshold[i]):
            signals[i] = 0.20
        
        # Short: 4h EMA20 falling, 1h high touches EMA1H from above, volume spike
        elif (ema_4h_slope[i] < 0 and 
              high[i] >= ema_1h[i] and 
              low[i] < ema_1h[i] and  # crossed below
              volume[i] > vol_threshold[i]):
            signals[i] = -0.20
        
        # Exit: 4h trend changes or price moves away from EMA1H
        elif (i > 0 and 
              ((signals[i-1] == 0.20 and (ema_4h_slope[i] <= 0 or low[i] > ema_1h[i] * 1.005)) or
               (signals[i-1] == -0.20 and (ema_4h_slope[i] >= 0 or high[i] < ema_1h[i] * 0.995)))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1h_EMA20_Pullback_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0