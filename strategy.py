#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian Channel Breakout with Volume Confirmation and ATR Stoploss.
# Uses weekly trend filter (EMA50) to align with long-term direction and avoid counter-trend trades.
# Entry: Price breaks above/below 20-period Donchian channel on 12h timeframe with volume > 1.5x 20-period average.
# Exit: Opposite Donchian band touch or ATR-based stoploss (2x ATR).
# Works in bull/bear markets via trend filter and volatility-based stops.
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
    
    # Weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50 = np.full(len(close_1w), np.nan)
    for i in range(49, len(close_1w)):
        ema_50[i] = np.mean(close_1w[i-49:i+1])  # Simple MA for efficiency
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Donchian channels (20-period) on 12h
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(19, n):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: only trade in direction of weekly EMA50
        trend_up = close[i] > ema_50_aligned[i]
        trend_down = close[i] < ema_50_aligned[i]
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price touches lower Donchian band or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.0 * atr_approx
            
            if (close[i] <= lowest_low[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price touches upper Donchian band or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.0 * atr_approx
            
            if (close[i] >= highest_high[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume and trend confirmation
            if volume_filter:
                # Buy breakout above upper Donchian band in uptrend
                if (close[i] > highest_high[i] and close[i-1] <= highest_high[i] and trend_up):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Sell breakdown below lower Donchian band in downtrend
                elif (close[i] < lowest_low[i] and close[i-1] >= lowest_low[i] and trend_down):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals