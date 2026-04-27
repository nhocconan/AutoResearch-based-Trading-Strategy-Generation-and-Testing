#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for trend direction and market regime
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h EMA 50 for trend direction
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Calculate 12h EMA 200 for long-term trend filter
    ema_200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 12h EMAs to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_200_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    # Calculate 12h ADX for trend strength (14-period)
    tr1 = np.abs(high_12h - low_12h)
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    up_move = np.diff(high_12h, prepend=np.nan)
    down_move = -np.diff(low_12h, prepend=np.nan)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_12h = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Volume filter: 6h volume > 1.8x 20-period average (more stringent)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine 12h trend regime
        uptrend = ema_50_aligned[i] > ema_200_aligned[i]
        strong_trend = adx_aligned[i] > 25
        
        # Long conditions: uptrend + strong trend + price above EMA50 + volume spike
        if (uptrend and strong_trend and 
            close[i] > ema_50_aligned[i] and 
            volume_spike[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: downtrend + strong trend + price below EMA50 + volume spike
        elif (not uptrend and strong_trend and 
              close[i] < ema_50_aligned[i] and 
              volume_spike[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend weakening or price crosses EMA200
        elif position == 1 and (not uptrend or close[i] < ema_200_aligned[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (uptrend or close[i] > ema_200_aligned[i]):
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

name = "6h_EMA50_200_ADX25_VolumeSpike_12h_v1"
timeframe = "6h"
leverage = 1.0