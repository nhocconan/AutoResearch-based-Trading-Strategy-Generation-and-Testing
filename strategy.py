#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d Williams %R momentum and 1w EMA134 trend alignment.
# Enters long when price > 1w EMA134 and Williams %R < -80 (oversold).
# Enters short when price < 1w EMA134 and Williams %R > -20 (overbought).
# Uses volume confirmation (>1.5x 20-period average) and 08-20 UTC session filter.
# Targets 12-37 trades/year (50-150 total over 4 years) with strict entry conditions.
# Williams %R captures mean reversion in ranging markets while following weekly trend.
# Works in bull/bear by aligning with higher timeframe trend and fading extremes.

name = "12h_1w_EMA134_1d_WilliamsR_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for EMA134 trend (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_134_1w = pd.Series(close_1w).ewm(span=134, adjust=False, min_periods=134).mean().values
    ema_134_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_134_1w)
    
    # Get 1d data for Williams %R (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure enough data for all indicators (Williams %R needs 14, EMA needs 134)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_134_1w_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(volume_ma[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above 1w EMA134 AND Williams %R oversold (< -80) with volume
            if (close[i] > ema_134_1w_aligned[i] and 
                williams_r_aligned[i] < -80 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below 1w EMA134 AND Williams %R overbought (> -20) with volume
            elif (close[i] < ema_134_1w_aligned[i] and 
                  williams_r_aligned[i] > -20 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses below 1w EMA134 or Williams %R becomes overbought
            if close[i] < ema_134_1w_aligned[i] or williams_r_aligned[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above 1w EMA134 or Williams %R becomes oversold
            if close[i] > ema_134_1w_aligned[i] or williams_r_aligned[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals