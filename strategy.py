#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian Channel Breakout with Weekly Trend Filter and Volume Confirmation.
# Uses daily Donchian(20) for breakout signals, weekly EMA(13) for trend direction,
# and volume > 1.5x 20-day average for confirmation. Designed for 1d timeframe
# to capture multi-day trends while minimizing trade frequency and fee drag.
# Works in both bull and bear markets via trend-filtered breakouts.

name = "1d_donchian20_weekly_trend_vol_v1"
timeframe = "1d"
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
    
    # Weekly EMA for trend filter (13-period)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(13) on weekly data
    ema_13_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 13:
        ema_13_1w[12] = np.mean(close_1w[:13])
        for i in range(13, len(close_1w)):
            ema_13_1w[i] = (close_1w[i] * 2/14) + (ema_13_1w[i-1] * 12/14)
    
    # Align weekly EMA to daily timeframe (shifted by 1 weekly bar)
    ema_13_aligned = align_htf_to_ltf(prices, df_1w, ema_13_1w)
    
    # Daily Donchian Channel (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_13_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price reaches Donchian low or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.5 * atr_approx
            
            if (close[i] <= donchian_low[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches Donchian high or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.5 * atr_approx
            
            if (close[i] >= donchian_high[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend filter
            if volume_filter:
                # Long breakout: price breaks above Donchian high in uptrend
                if (close[i] > donchian_high[i] and close[i-1] <= donchian_high[i] and
                    close[i] > ema_13_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short breakdown: price breaks below Donchian low in downtrend
                elif (close[i] < donchian_low[i] and close[i-1] >= donchian_low[i] and
                      close[i] < ema_13_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals