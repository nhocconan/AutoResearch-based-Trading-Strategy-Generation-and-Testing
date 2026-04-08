#!/usr/bin/env python3
"""
1d KAMA Trend with 1w Trend Filter and Volume Confirmation
Hypothesis: KAMA adapts to market noise - in trending markets it tracks price closely,
while in ranging markets it stays flat. Combined with 1w trend filter to ensure we
only trade in the direction of the higher timeframe trend, and volume confirmation
to avoid false signals. Works in bull via longs, in bear via shorts.
Target: 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_trend_1w_volume_v1"
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
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # KAMA (Adaptive Moving Average) parameters
    er_length = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, k=er_length))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Handle volatility calculation properly
    volatility_temp = np.abs(np.diff(close))
    volatility = np.convolve(volatility_temp, np.ones(er_length), mode='same')
    volatility[:er_length-1] = np.nan
    volatility[er_length-1:] = np.sum(np.abs(np.diff(close[:er_length+1])), axis=0) if len(change) > 0 else np.full_like(change, np.nan)
    # Simpler approach
    volatility = pd.Series(close).rolling(window=er_length).apply(lambda x: np.sum(np.abs(np.diff(x))), raw=True).values
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1))**2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[er_length] = close[er_length]  # Start with first valid value
    for i in range(er_length + 1, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(er_length, 20, 50), n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below KAMA or trend reverses
            if close[i] < kama[i] or close[i] < ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above KAMA or trend reverses
            if close[i] > kama[i] or close[i] > ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter: price vs 1w EMA50
            uptrend = close[i] > ema_50_1w_aligned[i]
            downtrend = close[i] < ema_50_1w_aligned[i]
            
            # Long: price above KAMA + uptrend + volume spike
            if (close[i] > kama[i] and 
                uptrend and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: price below KAMA + downtrend + volume spike
            elif (close[i] < kama[i] and 
                  downtrend and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals