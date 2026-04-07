#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout with 1w trend filter and volume confirmation
# Uses daily Donchian(20) breakouts for trend continuation, filtered by weekly EMA50 trend
# Volume confirmation avoids false breakouts
# Designed for low frequency (target: 10-25 trades/year) to minimize fee impact
# Works in both bull/bear via trend filter: only trade in direction of weekly trend

name = "1d_donchian20_weekly_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d Donchian(20) channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Trend filter from 1w EMA
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Entry and exit logic
        if position == 0:  # Flat, look for entry
            # Long breakout above upper Donchian with uptrend and volume
            if (high[i] > high_roll[i-1]) and uptrend and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Short breakdown below lower Donchian with downtrend and volume
            elif (low[i] < low_roll[i-1]) and downtrend and vol_confirm:
                position = -1
                signals[i] = -0.25
        elif position == 1:  # Long position
            # Exit on breakdown below lower Donchian or trend reversal
            if (low[i] < low_roll[i-1]) or (close[i] < ema_50_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit on breakout above upper Donchian or trend reversal
            if (high[i] > high_roll[i-1]) or (close[i] > ema_50_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
    
    return signals