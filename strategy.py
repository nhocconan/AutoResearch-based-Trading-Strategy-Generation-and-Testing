#!/usr/bin/env python3
"""
1h 4h/1d Confluence Strategy with Volume Spike and Session Filter
Hypothesis: Use 4h trend (EMA21) and 1d momentum (RSI14) for directional bias,
            enter on 1h breakouts with volume spike during active session (08-20 UTC).
            Designed for low trade frequency (15-37/year) with high edge in both bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    # Get 1d data for momentum filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA21 for trend
    ema_21_4h = pd.Series(df_4h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    # Calculate 1d RSI14 for momentum
    delta = pd.Series(df_1d['close']).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi_14_1d = (100 - (100 / (1 + rs))).values
    
    # Align HTF indicators to 1h timeframe
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # 1h Donchian breakout channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100  # need enough history for calculations
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_21_4h_aligned[i]) or 
            np.isnan(rsi_14_1d_aligned[i]) or
            np.isnan(high_20[i]) or
            np.isnan(low_20[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_21 = ema_21_4h_aligned[i]
        rsi_14 = rsi_14_1d_aligned[i]
        
        if position == 0:
            # Look for long setup: uptrend + bullish momentum + breakout + volume + session
            if (price > ema_21 and  # 4h uptrend
                rsi_14 > 50 and     # 1d bullish momentum
                price > high_20[i] and  # 1h breakout above 20-period high
                volume_spike[i] and     # volume confirmation
                in_session[i]):         # active session
                signals[i] = 0.20
                position = 1
            # Look for short setup: downtrend + bearish momentum + breakdown + volume + session
            elif (price < ema_21 and    # 4h downtrend
                  rsi_14 < 50 and       # 1d bearish momentum
                  price < low_20[i] and # 1h breakdown below 20-period low
                  volume_spike[i] and   # volume confirmation
                  in_session[i]):       # active session
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Hold long position
            signals[i] = 0.20
            # Exit: breakdown below 20-period low or trend/momentum deterioration
            if (price < low_20[i] or 
                price < ema_21 or 
                rsi_14 < 40):
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Hold short position
            signals[i] = -0.20
            # Exit: breakout above 20-period high or trend/momentum deterioration
            if (price > high_20[i] or 
                price > ema_21 or 
                rsi_14 > 60):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_4h1d_Confluence_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0