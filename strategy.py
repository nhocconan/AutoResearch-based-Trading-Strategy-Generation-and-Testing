#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_williams_alligator_ema200"
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
    
    # Williams Alligator from 12h
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    jaw = pd.Series(close_12h).rolling(window=13, center=False).mean().shift(8).values
    teeth = pd.Series(close_12h).rolling(window=8, center=False).mean().shift(5).values
    lips = pd.Series(close_12h).rolling(window=5, center=False).mean().shift(3).values
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # EMA200 filter from 1d
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume filter: current volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_200_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Alligator alignment: jaws < teeth < lips for downtrend, jaws > teeth > lips for uptrend
        alligator_long = jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i]
        alligator_short = jaw_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < lips_aligned[i]
        
        # EMA200 filter: price above/below long-term trend
        price_above_ema200 = close[i] > ema_200_aligned[i]
        price_below_ema200 = close[i] < ema_200_aligned[i]
        
        # Volume confirmation
        vol_ok = volume_filter[i]
        
        # Entry signals
        long_signal = alligator_long and price_above_ema200 and vol_ok
        short_signal = alligator_short and price_below_ema200 and vol_ok
        
        # Exit when Alligator lines cross (trend change)
        exit_long = jaw_aligned[i] < teeth_aligned[i]
        exit_short = jaw_aligned[i] > teeth_aligned[i]
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals