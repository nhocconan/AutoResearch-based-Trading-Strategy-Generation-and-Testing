#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1d Williams %R for overbought/oversold and 1w EMA34 for trend filter.
# Enters only during 08-20 UTC session. Uses mean reversion in ranging markets (Williams %R extremes)
# and trend following in trending markets (price relative to EMA34). Volume confirmation reduces false signals.
# Designed to work in both bull (trend following) and bear (mean reversion) markets.
name = "6h_1w_EMA34_1d_WilliamsR_Volume"
timeframe = "6h"
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
    
    # Get 1w data for EMA34 trend (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data for Williams %R (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close_1d) / (highest_high - lowest_low)) * -100,
        -50  # neutral when range is zero
    )
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(volume_ma[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long conditions: oversold (Williams %R < -80) AND price above 1w EMA34 (uptrend filter)
            # OR: Williams %R < -90 (extreme oversold) regardless of trend (mean reversion)
            if ((williams_r_aligned[i] < -80 and close[i] > ema_34_1w_aligned[i]) or
                (williams_r_aligned[i] < -90)) and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short conditions: overbought (Williams %R > -20) AND price below 1w EMA34 (downtrend filter)
            # OR: Williams %R > -10 (extreme overbought) regardless of trend (mean reversion)
            elif ((williams_r_aligned[i] > -20 and close[i] < ema_34_1w_aligned[i]) or
                  (williams_r_aligned[i] > -10)) and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long exit: price crosses below 1w EMA34 OR Williams %R returns to overbought (> -20)
            if close[i] < ema_34_1w_aligned[i] or williams_r_aligned[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short exit: price crosses above 1w EMA34 OR Williams %R returns to oversold (< -80)
            if close[i] > ema_34_1w_aligned[i] or williams_r_aligned[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals