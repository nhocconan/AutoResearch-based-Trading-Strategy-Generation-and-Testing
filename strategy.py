#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray Power combination with 1d trend filter
# Williams Alligator (jaw/teeth/lips) identifies trend direction and strength
# Elder Ray (Bull/Bear Power) measures momentum behind price moves
# 1d EMA34 filter ensures we only trade in alignment with higher timeframe trend
# Volume confirmation (>1.5x average) filters weak breakouts
# Discrete sizing 0.25 to minimize fee churn. Target: 50-150 trades over 4 years.
# Primary timeframe: 6h, HTF: 1d for EMA34 trend filter

name = "6h_WilliamsAlligator_ElderRay_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator from 6h data
    # Jaw: 13-period SMMA shifted 8 bars ahead
    # Teeth: 8-period SMMA shifted 5 bars ahead  
    # Lips: 5-period SMMA shifted 3 bars ahead
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Elder Ray Power (1-period EMA)
    bull_power = high - pd.Series(close).ewm(span=1, adjust=False).mean().values
    bear_power = low - pd.Series(close).ewm(span=1, adjust=False).mean().values
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Alligator signals: Mouth open (teeth above/below lips) indicates trend
        # Jaw below teeth/lips = uptrend, Jaw above teeth/lips = downtrend
        alligator_long = jaw[i] < teeth[i] and teeth[i] < lips[i]
        alligator_short = jaw[i] > teeth[i] and teeth[i] > lips[i]
        
        # Elder Ray: Positive bull power = buying pressure, negative bear power = selling pressure
        elder_long = bull_power[i] > 0
        elder_short = bear_power[i] < 0
        
        if position == 0:  # Flat - look for new entries
            # Long: Alligator uptrend + Elder Ray bull power + price > 1d EMA34 + volume spike
            if (alligator_long and elder_long and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator downtrend + Elder Ray bear power + price < 1d EMA34 + volume spike
            elif (alligator_short and elder_short and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator trend reverses OR Elder Ray turns negative OR price < 1d EMA34
            if (not alligator_long or not elder_long or 
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator trend reverses OR Elder Ray turns positive OR price > 1d EMA34
            if (not alligator_short or not elder_short or 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals