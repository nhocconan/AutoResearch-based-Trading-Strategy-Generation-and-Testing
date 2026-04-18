#!/usr/bin/env python3
"""
12h_1w_Donchian_Breakout_Trend_Filter
Hypothesis: Uses weekly Donchian(20) breakouts with 1d trend filter (EMA50) and volume confirmation.
In bull markets, buy upper band breakouts; in bear markets, sell lower band breakdowns.
Trades only when price is above/below EMA50 to avoid counter-trend whipsaws.
Targets 15-25 trades/year to minimize fee drag on 12h timeframe.
"""

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
    
    # Get weekly data for Donchian channels (20-period high/low)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Donchian channels on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 20-period rolling high/low on weekly data
    donchian_high = np.full(len(high_1w), np.nan)
    donchian_low = np.full(len(low_1w), np.nan)
    
    for i in range(20, len(high_1w)):
        donchian_high[i] = np.max(high_1w[i-20:i+1])
        donchian_low[i] = np.min(low_1w[i-20:i+1])
    
    # Align weekly Donchian levels to 12h timeframe
    dh_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    dh_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on daily close
    ema_50 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50[i] = (close_1d[i] * 2/51) + (ema_50[i-1] * (1 - 2/51))
    
    # Align daily EMA50 to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: current volume > 1.3 x 30-period average
    vol_ma = np.full(n, np.nan)
    for i in range(30, n):
        vol_ma[i] = np.mean(volume[i-30:i])
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 50)  # Need volume MA and EMA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(dh_high_aligned[i]) or np.isnan(dh_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above weekly Donchian high with volume confirmation and uptrend (price > EMA50)
            if (close[i] > dh_high_aligned[i] and vol_confirm[i] and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below weekly Donchian low with volume confirmation and downtrend (price < EMA50)
            elif (close[i] < dh_low_aligned[i] and vol_confirm[i] and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to weekly Donchian low or breaks below EMA50
            if close[i] < dh_low_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to weekly Donchian high or breaks above EMA50
            if close[i] > dh_high_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1w_Donchian_Breakout_Trend_Filter"
timeframe = "12h"
leverage = 1.0