#%%
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike.
Camarilla levels act as intraday support/resistance; breakout with volume confirms momentum.
1d EMA34 filters for trend direction to avoid counter-trend trades.
Volume spike (>1.5x average) ensures participation.
Designed to work in both bull (breakouts continue) and bear (fades at opposite levels) markets.
Target: 20-40 trades/year to avoid fee drag.
"""

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
    
    # Load daily data for Camarilla and EMA34 - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # HLC of previous day
    prev_high = df_daily['high'].shift(1).values
    prev_low = df_daily['low'].shift(1).values
    prev_close = df_daily['close'].shift(1).values
    
    # Camarilla R1, S1
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Calculate 1d EMA34 for trend filter
    close_series = pd.Series(df_daily['close'].values)
    ema34 = close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align Camarilla and EMA to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_daily, R1)
    S1_aligned = align_htf_to_ltf(prices, df_daily, S1)
    ema34_aligned = align_htf_to_ltf(prices, df_daily, ema34)
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume, above daily EMA34 (uptrend)
            if (close[i] > R1_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20[i] and
                close[i] > ema34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume, below daily EMA34 (downtrend)
            elif (close[i] < S1_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20[i] and
                  close[i] < ema34_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Camarilla level
            if position == 1:
                if close[i] < S1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > R1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4H_Camarilla_R1S1_Volume_EMA34_Trend"
timeframe = "4h"
leverage = 1.0
#%%