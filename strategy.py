#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R extremes + volume confirmation + trend filter
# Williams %R identifies overbought/oversold conditions on 1d timeframe
# Long when %R < -80 (oversold) with volume confirmation and price > 20-period EMA (uptrend filter)
# Short when %R > -20 (overbought) with volume confirmation and price < 20-period EMA (downtrend filter)
# Uses discrete position sizing 0.25 to target ~20-40 trades/year and minimize fee drag
# Works in bull/bear markets: mean reversion in ranges, trend filter avoids counter-trend trades

name = "6h_1d_williamsr_meanrev_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Replace division by zero with -50 (neutral)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align 1d Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Pre-compute 6h EMA(20) for trend filter
    close_s = pd.Series(close)
    ema_20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Pre-compute volume confirmation: current 6h volume > 1.5x average 6h volume (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_20[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long if Williams %R rises above -50 (exiting oversold) or trend turns down
            if williams_r_aligned[i] > -50 or close[i] < ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if Williams %R falls below -50 (exiting overbought) or trend turns up
            if williams_r_aligned[i] < -50 or close[i] > ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Mean reversion: enter at extremes with volume confirmation and trend alignment
            if (williams_r_aligned[i] < -80 and  # Oversold
                volume_confirmed[i] and 
                close[i] > ema_20[i]):  # Uptrend filter
                position = 1
                signals[i] = 0.25
            elif (williams_r_aligned[i] > -20 and  # Overbought
                  volume_confirmed[i] and 
                  close[i] < ema_20[i]):  # Downtrend filter
                position = -1
                signals[i] = -0.25
    
    return signals