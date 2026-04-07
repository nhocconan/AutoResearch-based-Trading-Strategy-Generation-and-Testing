#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h_4h1d_trend_follow_volume
# Hypothesis: Use 4h trend (EMA21) and 1d trend (EMA50) as directional filters, enter on 1h EMA(8/21) crossovers with volume confirmation.
# 4h/1d filters prevent counter-trend trades, reducing whipsaw in sideways markets. Volume confirms institutional participation.
# Target: 15-35 trades/year (~60-140 over 4 years) to minimize fee drag.
name = "1h_4h1d_trend_follow_volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # 4h EMA(21) for trend
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=21, adjust=False).mean().values
    ema_4h_1h = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema_1d_1h = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1h EMA(8) and EMA(21) for entry timing
    ema_8 = pd.Series(close).ewm(span=8, adjust=False).mean().values
    ema_21 = pd.Series(close).ewm(span=21, adjust=False).mean().values
    
    # Volume confirmation: 1h volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(21, n):
        # Skip if required data not available
        if (np.isnan(ema_4h_1h[i]) or np.isnan(ema_1d_1h[i]) or 
            np.isnan(ema_8[i]) or np.isnan(ema_21[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: EMA(8) crosses below EMA(21) OR trend turns bearish on 4h or 1d
            if ema_8[i] < ema_21[i] or close[i] < ema_4h_1h[i] or close[i] < ema_1d_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long position
        elif position == -1:  # Short position
            # Exit: EMA(8) crosses above EMA(21) OR trend turns bullish on 4h or 1d
            if ema_8[i] > ema_21[i] or close[i] > ema_4h_1h[i] or close[i] > ema_1d_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: EMA(8) crosses above EMA(21) with volume and bullish 4h/1d trend
            if ema_8[i] > ema_21[i] and ema_8[i-1] <= ema_21[i-1] and vol_confirm and close[i] > ema_4h_1h[i] and close[i] > ema_1d_1h[i]:
                position = 1
                signals[i] = 0.20
            # Enter short: EMA(8) crosses below EMA(21) with volume and bearish 4h/1d trend
            elif ema_8[i] < ema_21[i] and ema_8[i-1] >= ema_21[i-1] and vol_confirm and close[i] < ema_4h_1h[i] and close[i] < ema_1d_1h[i]:
                position = -1
                signals[i] = -0.20
    
    return signals