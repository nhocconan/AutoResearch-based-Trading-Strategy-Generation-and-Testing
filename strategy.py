#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h 20-period Donchian breakout with weekly trend filter and volume confirmation
# In bull markets: price breaks above 20-period high with weekly uptrend and volume
# In bear markets: price breaks below 20-period low with weekly downtrend and volume
# Weekly trend uses 50-period EMA to avoid whipsaw
# Volume confirms breakout strength
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for 20-period lookback and weekly EMA
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h Donchian channel (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # Weekly trend: 50-period EMA on weekly close
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean()
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(50, n):  # Start after weekly EMA warmup
        # Skip if any required data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(weekly_ema_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: price breaks above Donchian high + weekly uptrend + volume
        if (close[i] > donch_high[i] and 
            close[i] > weekly_ema_aligned[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: price breaks below Donchian low + weekly downtrend + volume
        elif (close[i] < donch_low[i] and 
              close[i] < weekly_ema_aligned[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: price crosses back to weekly EMA (trend reversal signal)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < weekly_ema_aligned[i]) or
               (signals[i-1] == -0.25 and close[i] > weekly_ema_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_Donchian20_WeeklyEMA_Volume"
timeframe = "12h"
leverage = 1.0