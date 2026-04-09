#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R + 1w EMA trend filter with volume confirmation
# In bull markets: long when Williams %R < -80 (oversold) and price > weekly EMA50
# In bear markets: short when Williams %R > -20 (overbought) and price < weekly EMA50
# Uses discrete position sizing 0.25 to limit trades and reduce fee drag
# Williams %R captures mean reversion swings; weekly EMA ensures trend alignment
# Volume confirmation filters weak breakouts

name = "1d_1w_williamsr_ema_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high_14 - lowest_low_14) != 0,
                          -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14),
                          -50)
    
    # Calculate 1w EMA(50)
    close_s_1w = pd.Series(close_1w)
    ema_50_1w = close_s_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d average volume (20-period)
    volume_s = pd.Series(volume)
    avg_volume_20 = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Align 1w indicators to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(avg_volume_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirmed = volume[i] > 1.5 * avg_volume_20[i]
        
        if position == 1:  # Long position
            # Exit long if Williams %R rises above -50 or price falls below weekly EMA
            if williams_r[i] > -50 or close[i] < ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit short if Williams %R falls below -50 or price rises above weekly EMA
            if williams_r[i] < -50 or close[i] > ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long when oversold in uptrend
            if williams_r[i] < -80 and close[i] > ema_50_1w_aligned[i] and volume_confirmed:
                position = 1
                signals[i] = 0.25
            # Enter short when overbought in downtrend
            elif williams_r[i] > -20 and close[i] < ema_50_1w_aligned[i] and volume_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals