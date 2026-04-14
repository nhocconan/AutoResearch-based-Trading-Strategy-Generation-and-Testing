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
    
    # Load 12h data (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h EMA(50)
    if len(close_12h) >= 50:
        ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False).values
    else:
        ema_50_12h = np.full_like(close_12h, np.nan)
    
    # Calculate 12h ATR(14) for volatility filter
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_12h[0] = tr_12h[0]
    
    atr_14_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 14:
        atr_14_12h[13] = np.mean(tr_12h[1:15])
        for i in range(15, len(close_12h)):
            atr_14_12h[i] = (atr_14_12h[i-1] * 13 + tr_12h[i]) / 14
    
    # Calculate 12h EMA(50) slope for trend direction
    ema_slope_12h = np.full_like(close_12h, np.nan)
    if len(ema_50_12h) >= 2:
        ema_slope_12h[1:] = ema_50_12h[1:] - ema_50_12h[:-1]
    
    # Align 12h indicators to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_slope_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_slope_12h)
    atr_14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h)
    
    # Calculate 6h Williams %R (14-period)
    highest_high_14 = np.full_like(close, np.nan)
    lowest_low_14 = np.full_like(close, np.nan)
    
    if len(close) >= 14:
        for i in range(13, len(close)):
            highest_high_14[i] = np.max(high[i-13:i+1])
            lowest_low_14[i] = np.min(low[i-13:i+1])
    
    williams_r = np.full_like(close, np.nan)
    for i in range(13, len(close)):
        if highest_high_14[i] != lowest_low_14[i]:
            williams_r[i] = (highest_high_14[i] - close[i]) / (highest_high_14[i] - lowest_low_14[i]) * -100
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(ema_slope_12h_aligned[i]) or 
            np.isnan(atr_14_12h_aligned[i]) or 
            np.isnan(williams_r[i])):
            signals[i] = 0.0
            continue
        
        # ATR filter: only trade when volatility is above average
        if atr_14_12h_aligned[i] <= 0:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price above 12h EMA50 with upward slope + Williams %R oversold (< -80)
            if (close[i] > ema_50_12h_aligned[i] and
                ema_slope_12h_aligned[i] > 0 and
                williams_r[i] < -80):
                position = 1
                signals[i] = position_size
            # Short: Price below 12h EMA50 with downward slope + Williams %R overbought (> -20)
            elif (close[i] < ema_50_12h_aligned[i] and
                  ema_slope_12h_aligned[i] < 0 and
                  williams_r[i] > -20):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price crosses below 12h EMA50 OR Williams %R overbought (> -20)
            if (close[i] < ema_50_12h_aligned[i] or 
                williams_r[i] > -20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price crosses above 12h EMA50 OR Williams %R oversold (< -80)
            if (close[i] > ema_50_12h_aligned[i] or 
                williams_r[i] < -80):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_12h_EMA50_WilliamsR"
timeframe = "6h"
leverage = 1.0