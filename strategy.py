#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d ADX trend filter and volume confirmation
# Williams Alligator identifies trend presence and direction via smoothed SMAs
# 1d ADX > 25 filters for trending markets to avoid chop
# Volume confirmation ensures breakout strength
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
name = "12h_WilliamsAlligator_1dADX_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d ADX for trend strength filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate ADX components on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0.0
    tr2[0] = 0.0
    tr3[0] = 0.0
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(high_1d)
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0
    for i in range(1, len(high_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0.0
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0.0
    
    # Smoothed ATR, +DM, -DM
    period = 14
    atr_1d = np.zeros_like(tr_1d)
    plus_dm_sm = np.zeros_like(plus_dm)
    minus_dm_sm = np.zeros_like(minus_dm)
    
    atr_1d[period-1] = tr_1d[:period].mean()
    plus_dm_sm[period-1] = plus_dm[:period].mean()
    minus_dm_sm[period-1] = minus_dm[:period].mean()
    
    for i in range(period, len(tr_1d)):
        atr_1d[i] = (atr_1d[i-1] * (period-1) + tr_1d[i]) / period
        plus_dm_sm[i] = (plus_dm_sm[i-1] * (period-1) + plus_dm[i]) / period
        minus_dm_sm[i] = (minus_dm_sm[i-1] * (period-1) + minus_dm[i]) / period
    
    # Directional Indicators
    plus_di_1d = 100 * plus_dm_sm / atr_1d
    minus_di_1d = 100 * minus_dm_sm / atr_1d
    
    # DX and ADX
    dx = np.zeros_like(plus_di_1d)
    for i in range(len(plus_di_1d)):
        if plus_di_1d[i] + minus_di_1d[i] != 0:
            dx[i] = 100 * np.abs(plus_di_1d[i] - minus_di_1d[i]) / (plus_di_1d[i] + minus_di_1d[i])
        else:
            dx[i] = 0.0
    
    adx_1d = np.zeros_like(dx)
    adx_1d[2*period-2] = dx[:2*period-1].mean() if len(dx) >= 2*period-1 else 0.0
    for i in range(2*period-1, len(dx)):
        adx_1d[i] = (adx_1d[i-1] * (period-1) + dx[i]) / period
    
    # Align ADX to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Williams Alligator on 12h: SMAs with specific periods and shifts
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    # Jaw (Blue) - 13-period SMMA shifted 8 bars
    jaw = pd.Series(close).rolling(window=jaw_period, min_periods=jaw_period).mean()
    jaw = jaw.shift(8)  # shift 8 bars forward
    
    # Teeth (Red) - 8-period SMMA shifted 5 bars
    teeth = pd.Series(close).rolling(window=teeth_period, min_periods=teeth_period).mean()
    teeth = teeth.shift(5)  # shift 5 bars forward
    
    # Lips (Green) - 5-period SMMA shifted 3 bars
    lips = pd.Series(close).rolling(window=lips_period, min_periods=lips_period).mean()
    lips = lips.shift(3)  # shift 3 bars forward
    
    # Convert to numpy arrays, handling NaN from shift
    jaw = jaw.values
    teeth = teeth.values
    lips = lips.values
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(jaw_period + 8, teeth_period + 5, lips_period + 3, 20)  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Alligator signals: 
        # Lips > Teeth > Jaw = strong uptrend (all aligned upward)
        # Lips < Teeth < Jaw = strong downtrend (all aligned downward)
        if position == 0:
            # Long: Bullish alignment + ADX > 25 + volume confirmation
            if (lips[i] > teeth[i] > jaw[i] and 
                adx_1d_aligned[i] > 25 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + ADX > 25 + volume confirmation
            elif (lips[i] < teeth[i] < jaw[i] and 
                  adx_1d_aligned[i] > 25 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if bearish alignment forms or ADX weakens
            if (lips[i] < teeth[i] < jaw[i]) or (adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if bullish alignment forms or ADX weakens
            if (lips[i] > teeth[i] > jaw[i]) or (adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals