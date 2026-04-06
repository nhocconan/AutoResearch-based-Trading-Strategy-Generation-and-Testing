#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian Channel Breakout with Weekly Trend Filter and Volume Confirmation.
# Uses weekly EMA(50) for trend direction and daily Donchian(20) breakouts for entries.
# Volume filter (current volume > 1.5x 20-period average) ensures quality signals.
# Works in bull markets via breakouts and in bear markets via short breakdowns.
# Target: 50-150 trades over 4 years (12-37/year).

name = "12h_donchian20_weekly_ema50_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(50)
    ema_50 = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema_50[49] = np.mean(close_1w[:50])
        for i in range(50, len(close_1w)):
            ema_50[i] = (close_1w[i] * 2/51) + (ema_50[i-1] * 49/51)
    
    # Align weekly EMA to 12h timeframe (shifted by 1 weekly bar)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Daily Donchian(20) for entry signals
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily Donchian channels
    upper = np.full(len(high_1d), np.nan)
    lower = np.full(len(low_1d), np.nan)
    
    for i in range(19, len(high_1d)):
        upper[i] = np.max(high_1d[i-19:i+1])
        lower[i] = np.min(low_1d[i-19:i+1])
    
    # Align Donchian levels to 12h timeframe (shifted by 1 daily bar)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Trend filter: price above/below weekly EMA(50)
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price reaches lower Donchian or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.5 * atr_approx
            
            if (close[i] <= lower_aligned[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches upper Donchian or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.5 * atr_approx
            
            if (close[i] >= upper_aligned[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume and trend confirmation
            if volume_filter:
                # Long: breakout above upper Donchian in uptrend
                if (uptrend and 
                    close[i] > upper_aligned[i] and 
                    close[i-1] <= upper_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakdown below lower Donchian in downtrend
                elif (downtrend and 
                      close[i] < lower_aligned[i] and 
                      close[i-1] >= lower_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals