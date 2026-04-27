#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for 200-period EMA and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 200-period EMA on weekly close
    ema_200w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Calculate 14-period ADX on weekly data
    # True Range
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.diff(high_1w, prepend=np.nan)
    down_move = -np.diff(low_1w, prepend=np.nan)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr_w = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr_w
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr_w
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_w = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align weekly indicators to daily timeframe
    ema_200w_aligned = align_htf_to_ltf(prices, df_1w, ema_200w)
    adx_w_aligned = align_htf_to_ltf(prices, df_1w, adx_w)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200w_aligned[i]) or np.isnan(adx_w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price above weekly EMA200, ADX > 25, volume spike
        if (close[i] > ema_200w_aligned[i] and 
            adx_w_aligned[i] > 25 and 
            volume_spike[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price below weekly EMA200, ADX > 25, volume spike
        elif (close[i] < ema_200w_aligned[i] and 
              adx_w_aligned[i] > 25 and 
              volume_spike[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price crosses back over weekly EMA200
        elif position == 1 and close[i] < ema_200w_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > ema_200w_aligned[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_WeeklyEMA200_ADX25_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0