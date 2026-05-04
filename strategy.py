#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R extremes with 1w EMA50 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions on daily timeframe.
# Enters long when %R crosses above -80 from below with volume spike and 1w EMA50 uptrend.
# Enters short when %R crosses below -20 from above with volume spike and 1w EMA50 downtrend.
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for 7-25 trades/year on 1d timeframe.
# Works in bull markets via mean reversion from oversold and in bear markets via mean reversion from overbought.
# 1w EMA50 provides robust trend filter that avoids whipsaws during sideways markets.

name = "1d_WilliamsR_Extremes_1wEMA50_Trend_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Williams %R and EMA50 - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Williams %R (14-period) on 1w data
    highest_high_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1w) / (highest_high_14 - lowest_low_14)
    
    # Align Williams %R to 1d timeframe (wait for completed 1w bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    
    # Get 1w data for EMA50 trend filter - ONCE before loop
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 1d timeframe (wait for completed 1w bar)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate volume spike filter (20-period volume MA)
    vol_ma_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R crosses above -80 from below AND volume spike AND 1w EMA50 uptrend
            if (williams_r_aligned[i] > -80 and 
                williams_r_aligned[i-1] <= -80 and  # crossed above -80 from below
                volume_spike[i] and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R crosses below -20 from above AND volume spike AND 1w EMA50 downtrend
            elif (williams_r_aligned[i] < -20 and 
                  williams_r_aligned[i-1] >= -20 and  # crossed below -20 from above
                  volume_spike[i] and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -20 (overbought) OR trend reverses
            if williams_r_aligned[i] >= -20 or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -80 (oversold) OR trend reverses
            if williams_r_aligned[i] <= -80 or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals