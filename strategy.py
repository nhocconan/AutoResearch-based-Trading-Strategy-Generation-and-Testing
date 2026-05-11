#!/usr/bin/env python3
name = "1d_WeeklyVolatilityBreakout"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for volatility and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly ATR for volatility breakout
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Weekly high/low for breakout levels (using previous week's data)
    prev_high_1w = np.concatenate([[np.nan], high_1w[:-1]])
    prev_low_1w = np.concatenate([[np.nan], low_1w[:-1]])
    breakout_up = prev_high_1w + 0.5 * atr_1w  # Break above prev week high + 0.5*ATR
    breakout_down = prev_low_1w - 0.5 * atr_1w  # Break below prev week low - 0.5*ATR
    
    # Daily trend filter: price above/below 50-day EMA
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # Align weekly indicators to daily
    breakout_up_aligned = align_htf_to_ltf(prices, df_1w, breakout_up)
    breakout_down_aligned = align_htf_to_ltf(prices, df_1w, breakout_down)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)  # EMA50 and vol MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(breakout_up_aligned[i]) or np.isnan(breakout_down_aligned[i]) or 
            np.isnan(atr_1w_aligned[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly breakout level with volume confirmation and above EMA50
            if (close[i] > breakout_up_aligned[i] and 
                vol_confirm[i] and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly breakout level with volume confirmation and below EMA50
            elif (close[i] < breakout_down_aligned[i] and 
                  vol_confirm[i] and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price closes below breakout level or EMA50
            if close[i] < breakout_up_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price closes above breakout level or EMA50
            if close[i] > breakout_down_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals