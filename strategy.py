#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d trend filter and volume confirmation.
# Long when price breaks above 20-period upper Donchian with bullish 1d EMA trend and volume spike.
# Short when price breaks below 20-period lower Donchian with bearish 1d EMA trend and volume spike.
# Exit when price returns to middle Donchian (mean reversion).
# Uses 12h timeframe for lower frequency trading to minimize fee drag while capturing strong trends.
# Target: 12-37 trades per year (50-150 total over 4 years) to avoid fee drag.
# Trend filter uses 1d EMA50 to ensure alignment with higher timeframe trend.
# Volume filter requires 1.5x average volume to confirm breakout strength.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1-day EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Donchian Channel (20-period) on 12h timeframe
    donch_period = 20
    upper_donch = pd.Series(high).rolling(window=donch_period, min_periods=donch_period).max().values
    lower_donch = pd.Series(low).rolling(window=donch_period, min_periods=donch_period).min().values
    middle_donch = (upper_donch + lower_donch) / 2.0
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(upper_donch[i]) or 
            np.isnan(lower_donch[i]) or np.isnan(middle_donch[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above upper Donchian, above 1d EMA50, volume spike
        if (close[i] > upper_donch[i] and 
            close[i] > ema50_1d_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price breaks below lower Donchian, below 1d EMA50, volume spike
        elif (close[i] < lower_donch[i] and 
              close[i] < ema50_1d_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to middle Donchian (mean reversion)
        elif position == 1 and close[i] < middle_donch[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > middle_donch[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Donchian20_1dEMA50_VolumeFilter"
timeframe = "12h"
leverage = 1.0