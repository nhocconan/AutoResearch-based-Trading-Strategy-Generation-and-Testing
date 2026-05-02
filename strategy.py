#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + 1w EMA50 trend filter + volume confirmation
# Uses 1d primary timeframe for signal generation with Williams Alligator (jaw/teeth/lips)
# 1w EMA50 trend filter provides higher timeframe bias (price > EMA50 for longs, < for shorts)
# Volume confirmation (1.5x 20-period average) filters for strong participation
# Williams Alligator gives clear trend signals: lips above teeth above jaw = uptrend
# Discrete position sizing (0.25) balances profit potential with fee drag minimization
# Target: 30-80 total trades over 4 years (7-20/year) for 1d timeframe
# Works in both bull and bear markets by only trading in direction of 1w trend

name = "1d_WilliamsAlligator_1wEMA50_Trend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Williams Alligator on 1d timeframe
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward  
    # Lips: 5-period SMMA, shifted 3 bars forward
    # Using SMA as approximation for SMMA (simple moving average)
    jaw = pd.Series(high).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(low).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Bullish Alligator: lips > teeth > jaw + volume spike + price > 1w EMA50
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and volume_spike[i] and close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Bearish Alligator: lips < teeth < jaw + volume spike + price < 1w EMA50
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and volume_spike[i] and close[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator turns bearish (lips < teeth) or price < 1w EMA50
            if lips[i] < teeth[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator turns bullish (lips > teeth) or price > 1w EMA50
            if lips[i] > teeth[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals