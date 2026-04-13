#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d trend filter.
# Long: Price breaks above Donchian(20) high + volume > 1.5x 20-period average + 1d close > 1d EMA(50)
# Short: Price breaks below Donchian(20) low + volume > 1.5x 20-period average + 1d close < 1d EMA(50)
# Exit: Opposite Donchian break or trailing stop via signal=0 when adverse move > 2*ATR(20)
# Uses volume to filter breakouts, 1d EMA for trend alignment, reduces false signals.
# Target: 80-150 total trades over 4 years (20-38/year) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Average volume (20-period) for confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # ATR(20) for volatility and trailing stop
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = np.full(n, np.nan)
    for i in range(20, n):
        atr[i] = np.mean(tr[i-20:i])
    
    # 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on 1d
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 / (50 + 1)) + (ema_50_1d[i-1] * (49 / (50 + 1)))
    
    # Align 1d EMA50 to 4h
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(atr[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        atr_val = atr[i]
        d_high = donchian_high[i]
        d_low = donchian_low[i]
        ema1d = ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        # Trend filter: 1d close > EMA50 for long, < EMA50 for short
        uptrend = price > ema1d
        downtrend = price < ema1d
        
        if position == 0:
            # Long: break above Donchian high + volume + uptrend
            if (price > d_high and volume_confirm and uptrend):
                position = 1
                entry_price = price
                signals[i] = position_size
            # Short: break below Donchian low + volume + downtrend
            elif (price < d_low and volume_confirm and downtrend):
                position = -1
                entry_price = price
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Trailing stop: exit if price drops > 2*ATR from entry
            if price < entry_price - 2.0 * atr_val:
                position = 0
                signals[i] = 0.0
            # Exit: price breaks below Donchian low
            elif price < d_low:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Trailing stop: exit if price rises > 2*ATR from entry
            if price > entry_price + 2.0 * atr_val:
                position = 0
                signals[i] = 0.0
            # Exit: price breaks above Donchian high
            elif price > d_high:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Donchian_Volume_Trend"
timeframe = "4h"
leverage = 1.0