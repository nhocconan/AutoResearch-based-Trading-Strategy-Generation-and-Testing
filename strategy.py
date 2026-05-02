#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + 1w trend filter + volume confirmation
# Williams Alligator identifies trend via three SMAs (Jaw=13, Teeth=8, Lips=5)
# In trending markets (1w close > 1w EMA50): trade Alligator alignment (bullish/bearish)
# In ranging markets (1w close < 1w EMA50): fade Alligator extremes
# Volume confirmation (1.5x 20-period average) ensures participation
# Discrete position sizing 0.25 balances risk and minimizes fee churn
# Targets 15-40 trades/year (60-160 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by adapting to regime via 1w EMA50

name = "1d_WilliamsAlligator_1wTrendRegime_VolumeConfirm_v1"
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
    
    # Load 1w data ONCE before loop for trend regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend regime
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    close_1w = df_1w['close'].values
    trend_aligned = align_htf_to_ltf(prices, df_1w, close_1w > ema50_1w)  # True if trending up
    
    # Calculate 1d Williams Alligator: Jaw (13), Teeth (8), Lips (5)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # Calculate 1d volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Alligator and volume MA)
    start_idx = 30  # max(20 for volume, 13 for Jaw) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(trend_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine regime from 1w trend
        trending_up = trend_aligned[i]  # True if 1w close > 1w EMA50
        ranging = not trending_up       # False if 1w close <= 1w EMA50
        
        if position == 0:  # Flat - look for new entries
            if trending_up:
                # In trending market: follow Alligator alignment (bullish: Lips > Teeth > Jaw)
                if (lips[i] > teeth[i] and teeth[i] > jaw[i] and
                    i > start_idx and 
                    not (lips[i-1] > teeth[i-1] and teeth[i-1] > jaw[i-1]) and  # Just aligned
                    volume_confirm[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: Alligator alignment bearish (Lips < Teeth < Jaw)
                elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and
                      i > start_idx and 
                      not (lips[i-1] < teeth[i-1] and teeth[i-1] < jaw[i-1]) and  # Just aligned
                      volume_confirm[i]):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:  # ranging market
                # In ranging market: fade Alligator extremes
                # Long: Lips crosses below Jaw (oversold bounce)
                if (lips[i] < jaw[i] and 
                    i > start_idx and lips[i-1] >= jaw[i-1] and  # Just crossed below
                    volume_confirm[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: Lips crosses above Jaw (overbought fade)
                elif (lips[i] > jaw[i] and 
                      i > start_idx and lips[i-1] <= jaw[i-1] and  # Just crossed above
                      volume_confirm[i]):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions
            exit_signal = False
            if trending_up:
                # Exit trending long when Alligator alignment breaks down (Lips <= Teeth)
                if lips[i] <= teeth[i]:
                    exit_signal = True
            else:
                # Exit ranging long when Lips rises above Jaw (weakening oversold)
                if lips[i] >= jaw[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions
            exit_signal = False
            if trending_up:
                # Exit trending short when Alligator alignment breaks down (Lips >= Teeth)
                if lips[i] >= teeth[i]:
                    exit_signal = True
            else:
                # Exit ranging short when Lips falls below Jaw (weakening overbought)
                if lips[i] <= jaw[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals