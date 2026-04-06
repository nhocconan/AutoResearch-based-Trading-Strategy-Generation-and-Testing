#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian Channel Breakout with Daily EMA Trend Filter and Volume Confirmation.
# Uses Donchian(20) breakouts on 12h timeframe, filtered by daily EMA(50) trend direction.
# Volume filter (current volume > 1.3x 20-period average) ensures quality signals.
# ATR-based stoploss (2.0 * ATR(14)) manages risk.
# Designed to work in both bull and bear markets by aligning with higher timeframe trend.
# Target: 50-150 trades over 4 years (12-37/year).

name = "12h_donchian20_1d_ema_vol_v1"
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
    
    # Daily EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on daily timeframe
    ema_50 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50[49] = close_1d[49]  # Simple average for first value
        multiplier = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema_50[i] = (close_1d[i] - ema_50[i-1]) * multiplier + ema_50[i-1]
    
    # Align daily EMA to 12h timeframe (shifted by 1 daily bar)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Donchian Channel (20-period) on 12h timeframe
    high_20 = np.full(n, np.nan)
    low_20 = np.full(n, np.nan)
    for i in range(19, n):
        high_20[i] = np.max(high[i-19:i+1])
        low_20[i] = np.min(low[i-19:i+1])
    
    # ATR(14) for stoploss and volatility filter
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr[13] = np.mean(tr[1:14])  # Simple average for first value
        for i in range(14, n):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14  # Wilder's smoothing
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price reaches Donchian low or stoploss
            stop_loss_level = entry_price - 2.0 * atr[i]
            
            if (close[i] <= low_20[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches Donchian high or stoploss
            stop_loss_level = entry_price + 2.0 * atr[i]
            
            if (close[i] >= high_20[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend filter
            if volume_filter:
                # Long breakout: price breaks above Donchian high with uptrend
                if (close[i] > high_20[i] and close[i-1] <= high_20[i] and 
                    close[i] > ema_50_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short breakdown: price breaks below Donchian low with downtrend
                elif (close[i] < low_20[i] and close[i-1] >= low_20[i] and 
                      close[i] < ema_50_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals