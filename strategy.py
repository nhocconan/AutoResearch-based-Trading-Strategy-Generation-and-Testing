#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX(14) + Williams Alligator (Jaw/Teeth/Lips) with 12h EMA50 trend filter
# ADX > 25 identifies trending markets. Williams Alligator confirms trend direction:
#   Lips (5) > Teeth (8) > Jaw (13) = uptrend, reverse for downtrend.
# 12h EMA50 acts as higher timeframe trend filter to avoid counter-trend trades.
# Volume spike (volume > 1.5x 20-bar EMA) confirms breakout conviction.
# Designed for low trade frequency (12-30/year) to minimize fee drag while capturing
# strong trends in both bull and bear markets via clear trend/adx/alligator alignment.

name = "6h_ADX_Alligator_12hEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ADX(14)
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after sufficient warmup for ADX
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_12h_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate ADX components using data up to current bar
        lookback = min(14, i+1)
        if lookback < 14:
            # Not enough data for ADX calculation
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # True Range
        tr1 = high[i-lookback+1:i+1] - low[i-lookback+1:i+1]
        tr2 = np.abs(high[i-lookback+1:i+1] - close[i-lookback:i])
        tr3 = np.abs(low[i-lookback+1:i+1] - close[i-lookback:i])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.mean(tr)
        
        # Directional Movement
        up_move = high[i-lookback+1:i+1] - high[i-lookback:i]
        down_move = low[i-lookback:i] - low[i-lookback+1:i+1]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed values
        tr_sum = np.sum(tr)
        plus_dm_sum = np.sum(plus_dm)
        minus_dm_sum = np.sum(minus_dm)
        
        if tr_sum == 0:
            dx = 0.0
        else:
            plus_di = 100 * (plus_dm_sum / tr_sum)
            minus_di = 100 * (minus_dm_sum / tr_sum)
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) != 0 else 0.0
        
        # Williams Alligator components (smoothed medians)
        jaw_period = 13
        teeth_period = 8
        lips_period = 5
        
        if i >= jaw_period:
            jaw_data = close[i-jaw_period+1:i+1]
            teeth_data = close[i-teeth_period+1:i+1]
            lips_data = close[i-lips_period+1:i+1]
            
            jaw = np.median(jaw_data)
            teeth = np.median(teeth_data)
            lips = np.median(lips_data)
        else:
            # Not enough data for Alligator
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: volume > 1.5x 20-bar EMA
        vol_lookback = min(20, i+1)
        vol_ema = pd.Series(volume[i-vol_lookback+1:i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1]
        volume_spike = volume[i] > (1.5 * vol_ema)
        
        # Alligator trend conditions
        alligator_long = lips > teeth > jaw
        alligator_short = lips < teeth < jaw
        
        # ADX trend strength condition
        strong_trend = dx > 25
        
        if position == 0:
            # Long: ADX > 25, Alligator aligned up, 12h EMA50 uptrend, volume spike
            if strong_trend and alligator_long and (ema_50_12h_aligned[i] > close[i]) and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25, Alligator aligned down, 12h EMA50 downtrend, volume spike
            elif strong_trend and alligator_short and (ema_50_12h_aligned[i] < close[i]) and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator loses alignment or ADX weakens
            if not (alligator_long and strong_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator loses alignment or ADX weakens
            if not (alligator_short and strong_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals