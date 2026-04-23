#!/usr/bin/env python3
"""
Hypothesis: 1h Williams %R Extreme + 4h EMA50 Trend Filter + Volume Spike + Session Filter (08-20 UTC)
- Williams %R(14) identifies overbought/oversold: long when < -80, short when > -20
- 4h EMA50 defines medium-term trend: only long when price > EMA50, short when price < EMA50
- Volume confirmation (> 2.0x 20-period average) filters weak signals
- Session filter (08-20 UTC) focuses on high-liquidity London/NY overlap
- Designed for 1h timeframe targeting 15-25 trades/year (60-100 over 4 years)
- Williams %R extremes provide high-probability mean reversion in ranging markets
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 1h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ma[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND price > 4h EMA50 AND volume spike
            if (williams_r[i] < -80 and 
                close[i] > ema_50_4h_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.20
                position = 1
            # Short: Williams %R > -20 (overbought) AND price < 4h EMA50 AND volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: Williams %R returns to neutral zone (-50) OR crosses 4h EMA50
            exit_signal = False
            if position == 1:
                # Exit long when Williams %R > -50 OR price < 4h EMA50
                if williams_r[i] > -50 or close[i] < ema_50_4h_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when Williams %R < -50 OR price > 4h EMA50
                if williams_r[i] < -50 or close[i] > ema_50_4h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_WilliamsR_Extreme_4hEMA50_Trend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0