#!/usr/bin/env python3
"""
1d_1w_Momentum_Breakout_Volume_Trend
Hypothesis: Weekly momentum combined with daily breakout and volume confirmation works across bull/bear cycles.
Uses weekly RSI for momentum regime and daily price action for precise entry. Weekly trend filter reduces false signals.
Designed for low trade frequency (10-25/year) to minimize fee drag while capturing major moves.
"""

name = "1d_1w_Momentum_Breakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly data for momentum and trend filters
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly RSI(14) for momentum regime
    delta = np.diff(close_1w)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close_1w, np.nan)
    avg_loss = np.full_like(close_1w, np.nan)
    
    if len(close_1w) >= 14:
        avg_gain[13] = np.mean(gain[0:14])
        avg_loss[13] = np.mean(loss[0:14])
        for i in range(14, len(close_1w)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = np.full_like(close_1w, np.nan)
    valid = (~np.isnan(avg_loss)) & (avg_loss != 0)
    rs[valid] = avg_gain[valid] / avg_loss[valid]
    
    rsi_1w = np.full_like(close_1w, np.nan)
    rsi_1w[valid] = 100 - (100 / (1 + rs[valid]))
    
    # Weekly EMA50 for trend filter
    ema_50_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[0:50])
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (ema_50_1w[i-1] * 49 + close_1w[i]) / 50
    
    # Daily Donchian(20) for breakout levels
    donch_high = np.full_like(high, np.nan)
    donch_low = np.full_like(low, np.nan)
    
    if len(high) >= 20:
        for i in range(19, len(high)):
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
    
    # Volume spike filter: current volume / 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    # Align weekly indicators to daily timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(20, 50)  # Ensure Donchian and weekly EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Enter long: price breaks above Donchian high AND weekly bullish (RSI>50) AND price>weekly EMA50 AND volume spike
            if (close[i] > donch_high[i] and 
                rsi_1w_aligned[i] > 50 and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Enter short: price breaks below Donchian low AND weekly bearish (RSI<50) AND price<weekly EMA50 AND volume spike
            elif (close[i] < donch_low[i] and 
                  rsi_1w_aligned[i] < 50 and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Minimum holding period: 3 days
            if bars_since_entry < 3:
                signals[i] = 0.25
            else:
                # Exit long: price breaks below Donchian low OR weekly momentum turns bearish (RSI<40)
                if close[i] < donch_low[i] or rsi_1w_aligned[i] < 40:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Minimum holding period: 3 days
            if bars_since_entry < 3:
                signals[i] = -0.25
            else:
                # Exit short: price breaks above Donchian high OR weekly momentum turns bullish (RSI>60)
                if close[i] > donch_high[i] or rsi_1w_aligned[i] > 60:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals