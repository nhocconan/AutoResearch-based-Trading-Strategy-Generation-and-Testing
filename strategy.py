#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams %R extreme reversals with 1w trend filter and volume confirmation
    # Williams %R identifies overbought/oversold conditions on 12h chart
    # 1w EMA filter ensures we only trade with the primary trend
    # Volume spike confirms institutional participation at extremes
    # Works in bull/bear by fading extremes in range and following trend in momentum
    # Target: 12-37 trades/year per symbol.
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate 1w EMA(34)
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Williams %R parameters
    williams_period = 14
    williams_overbought = -20
    williams_oversold = -80
    
    # Calculate Williams %R on 12h data
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(williams_period - 1, n):
        highest_high[i] = np.max(high[i-williams_period+1:i+1])
        lowest_low[i] = np.min(low[i-williams_period+1:i+1])
    
    williams_r = np.full(n, np.nan)
    for i in range(williams_period - 1, n):
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            williams_r[i] = -50  # neutral when range is zero
    
    # 12h volume spike filter (current volume > 2.0 * 20-period average)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from 1w EMA
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        # Williams %R signals with volume confirmation
        long_signal = williams_r[i] < williams_oversold and volume_spike[i]
        short_signal = williams_r[i] > williams_overbought and volume_spike[i]
        
        # Exit conditions: reverse signal or volume dropout
        long_exit = williams_r[i] > -50 or (not volume_spike[i])
        short_exit = williams_r[i] < -50 or (not volume_spike[i])
        
        if long_signal and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1w_williams_r_extreme_trend_vol_v1"
timeframe = "12h"
leverage = 1.0