#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily close above/below weekly EMA34 with volume confirmation captures
# strong trend continuation in both bull and bear markets. Weekly EMA34 acts as
# dynamic support/resistance; daily breakouts with volume indicate institutional
# participation. Low trade frequency (~15/year) avoids fee drag while capturing
# major moves. Works in bull (trend continuation) and bear (sharp reversals back
# to weekly trend).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for close price
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Get weekly data for EMA34
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA34
    ema_1w_34 = np.full(len(df_1w), np.nan)
    alpha = 2 / (34 + 1)
    for i in range(len(close_1w)):
        if i == 0:
            ema_1w_34[i] = close_1w[i]
        elif i < 34:
            ema_1w_34[i] = np.mean(close_1w[:i+1])
        else:
            if np.isnan(ema_1w_34[i-1]):
                ema_1w_34[i] = np.mean(close_1w[i-33:i+1])
            else:
                ema_1w_34[i] = close_1w[i] * alpha + ema_1w_34[i-1] * (1 - alpha)
    
    # Align weekly EMA34 to daily timeframe
    ema_1w_34_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_34)
    
    # Calculate 20-period volume average for confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Start after warmup period
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1w_34_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        
        # Volume confirmation: > 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long: daily close above weekly EMA34 with volume and prior close below
            if (volume_confirmation and 
                price > ema_1w_34_aligned[i] and 
                close[i-1] <= ema_1w_34_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: daily close below weekly EMA34 with volume and prior close above
            elif (volume_confirmation and 
                  price < ema_1w_34_aligned[i] and 
                  close[i-1] >= ema_1w_34_aligned[i-1]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: daily close below weekly EMA34
            if price < ema_1w_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: daily close above weekly EMA34
            if price > ema_1w_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1W_EMA34_Trend_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0