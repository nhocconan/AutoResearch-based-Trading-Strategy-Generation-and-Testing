#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian(20) Breakout with Daily EMA Filter and Volume Confirmation.
# Breakouts above/below 20-period 12h high/low, filtered by 1d EMA(50) for trend direction.
# Volume confirmation requires current volume > 1.8x 20-period average.
# ATR-based stoploss reduces risk. Designed for 12h timeframe to limit trade frequency.
# Works in bull/bear markets via EMA trend filter and breakout confirmation.

name = "12h_donchian20_1d_ema50_vol_v2"
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
    
    # Daily EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on daily close
    ema_50 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50[i] = (close_1d[i] * 0.03774) + (ema_50[i-1] * 0.96226)  # 2/(50+1)
    
    # Align EMA to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Donchian channels (20-period) on 12h data
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Volume filter: current volume > 1.8x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.8
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below Donchian low or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.5 * atr_approx
            
            if (close[i] < donchian_low[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.5 * atr_approx
            
            if (close[i] > donchian_high[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume and trend filter
            if volume_filter:
                # Long breakout above Donchian high with EMA uptrend
                if (close[i] > donchian_high[i] and close[i-1] <= donchian_high[i] and 
                    ema_50_aligned[i] > ema_50_aligned[i-1]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short breakdown below Donchian low with EMA downtrend
                elif (close[i] < donchian_low[i] and close[i-1] >= donchian_low[i] and 
                      ema_50_aligned[i] < ema_50_aligned[i-1]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals