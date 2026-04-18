#!/usr/bin/env python3
"""
12h_Supertrend_TrendFollowing_VolumeFilter
Hypothesis: Supertrend on 12h timeframe provides strong trend signals. Combined with volume confirmation and daily trend filter (EMA34), this strategy captures major moves while avoiding chop. Volume filter ensures momentum confirmation. Designed for 12h timeframe to target 12-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(10) for Supertrend
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Supertrend parameters
    atr_multiplier = 3.0
    upper_band = (high + low) / 2 + (atr_multiplier * atr)
    lower_band = (high + low) / 2 - (atr_multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.full(n, np.nan)
    direction = np.full(n, 1)  # 1 for uptrend, -1 for downtrend
    
    # Calculate Supertrend
    for i in range(1, n):
        # Upper and lower bands
        upper_band[i] = max(upper_band[i], upper_band[i-1]) if close[i-1] > supertrend[i-1] else upper_band[i]
        lower_band[i] = min(lower_band[i], lower_band[i-1]) if close[i-1] < supertrend[i-1] else lower_band[i]
        
        # Determine trend direction
        if close[i] > upper_band[i-1]:
            direction[i] = 1
        elif close[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        # Set Supertrend value
        supertrend[i] = lower_band[i] if direction[i] == 1 else upper_band[i]
    
    # Align daily EMA to 12h timeframe
    ema_34_aligned = ema_34_1d_aligned  # already aligned
    
    # Volume filter: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 30  # enough for Supertrend calculation
    
    for i in range(start_idx, n):
        if (np.isnan(supertrend[i]) or 
            np.isnan(ema_34_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        st_value = supertrend[i]
        ema_trend = ema_34_aligned[i]
        vol_filt = volume_filter[i]
        
        if position == 0:
            # Long: price above Supertrend (uptrend), price above daily EMA, volume confirmation
            if price > st_value and price > ema_trend and vol_filt:
                signals[i] = 0.25
                position = 1
            # Short: price below Supertrend (downtrend), price below daily EMA, volume confirmation
            elif price < st_value and price < ema_trend and vol_filt:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long: maintain position while uptrend continues and above daily EMA
            if price > st_value and price > ema_trend:
                signals[i] = 0.25
            else:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short: maintain position while downtrend continues and below daily EMA
            if price < st_value and price < ema_trend:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Supertrend_TrendFollowing_VolumeFilter"
timeframe = "12h"
leverage = 1.0