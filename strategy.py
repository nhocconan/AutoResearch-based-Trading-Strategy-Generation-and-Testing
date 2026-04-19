#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Williams Alligator with weekly EMA trend filter and volume confirmation
# Uses weekly EMA20 for stronger trend bias, reducing false signals in chop
# Williams Alligator identifies trend alignment: Lips>Teeth>Jaw for uptrend, Jaw>Teeth>Lips for downtrend
# Volume spike (>1.5x 20-period average) confirms momentum
# Target: 15-25 trades/year per symbol with disciplined entries
name = "1d_WilliamsAlligator_WeeklyEMA_Volume"
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
    
    # Weekly EMA20 for trend bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Williams Alligator components (SMMA = Smoothed Moving Average)
    def smoothed_moving_average(data, period):
        sma = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            sma[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                sma[i] = (sma[i-1] * (period-1) + data[i]) / period
        return sma
    
    # Calculate Alligator lines on daily data
    jaw = smoothed_moving_average(close, 13)  # Blue line (13-period)
    teeth = smoothed_moving_average(close, 8)   # Red line (8-period)
    lips = smoothed_moving_average(close, 5)    # Green line (5-period)
    
    # Volume spike: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + above weekly EMA + volume spike
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                close[i] > ema_20_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Jaw > Teeth > Lips (bearish alignment) + below weekly EMA + volume spike
            elif (jaw[i] > teeth[i] and teeth[i] > lips[i] and 
                  close[i] < ema_20_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if Alligator lines intertwine (Lips < Teeth) or price breaks below weekly EMA
            if (lips[i] < teeth[i]) or (close[i] < ema_20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if Alligator lines intertwine (Jaw < Teeth) or price breaks above weekly EMA
            if (jaw[i] < teeth[i]) or (close[i] > ema_20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals