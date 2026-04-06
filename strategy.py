#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian Breakout with Weekly Trend Filter and Volume Confirmation
# Uses daily Donchian channels (20-period) for breakout entries.
# Weekly EMA (50) determines trend direction: only long when above, short when below.
# Volume filter (current volume > 1.5x 20-day average) ensures quality signals.
# ATR-based stop loss (2x ATR) manages risk.
# Designed for low frequency (target 30-100 trades over 4 years) to minimize fee drag.
# Works in bull/bear markets via trend filter and breakout logic.

name = "1d_donchian20_weekly_ema_vol_v1"
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
    
    # Daily Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Weekly EMA (50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = np.full(len(close_1w), np.nan)
    for i in range(49, len(close_1w)):
        if i == 49:
            ema_1w[i] = np.mean(close_1w[:50])
        else:
            ema_1w[i] = close_1w[i] * 2 / (50 + 1) + ema_1w[i-1] * (49 / (50 + 1))
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            atr = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.0 * atr
            
            # Exit: stop loss or reverse signal
            if (close[i] < stop_loss_level or 
                close[i] < donchian_low[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            atr = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.0 * atr
            
            # Exit: stop loss or reverse signal
            if (close[i] > stop_loss_level or 
                close[i] > donchian_high[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend filter
            if volume_filter:
                # Long breakout: price breaks above Donchian high in uptrend
                if (close[i] > donchian_high[i] and 
                    close[i-1] <= donchian_high[i] and
                    close[i] > ema_1w_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short breakdown: price breaks below Donchian low in downtrend
                elif (close[i] < donchian_low[i] and 
                      close[i-1] >= donchian_low[i] and
                      close[i] < ema_1w_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals