#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R mean reversion with 12h trend filter and volume spike confirmation.
    # Williams %R identifies overbought/oversold conditions (long when < -80, short when > -20).
    # 12h EMA50 provides trend filter (long only when price > EMA50, short only when price < EMA50).
    # Volume spike (current > 1.5 * 20-period MA) confirms momentum behind the move.
    # Target: 50-150 total trades over 4 years = 12-37/year.
    # Works in bull markets (long oversold in uptrend) and bear markets (short overbought in downtrend).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 6h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate 6h volume MA(20) for spike confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike filter: current volume > 1.5 * 20-period MA
        volume_spike = volume[i] > 1.5 * volume_ma[i]
        
        # Williams %R conditions
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        # Trend filter from 12h EMA50
        uptrend = close[i] > ema_12h_aligned[i]
        downtrend = close[i] < ema_12h_aligned[i]
        
        # Entry conditions
        long_entry = oversold and uptrend and volume_spike
        short_entry = overbought and downtrend and volume_spike
        
        # Exit conditions: opposite Williams %R extreme or loss of trend
        long_exit = williams_r[i] > -20 or not uptrend
        short_exit = williams_r[i] < -80 or not downtrend
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
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

name = "6h_12h_williamsr_trend_volume_v1"
timeframe = "6h"
leverage = 1.0