#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with weekly trend filter and volume confirmation.
# Uses daily Donchian channel (20-day) breakouts for trend continuation.
# Weekly trend filter (price above/below 20-week EMA) ensures alignment with higher timeframe trend.
# Volume confirmation (current volume > 1.5x 20-day average) filters low-quality breakouts.
# Works in bull markets via upward breakouts and in bear markets via downward breakdowns.
# Target: 30-100 trades over 4 years (7-25/year).

name = "1d_donchian20_weekly_trend_vol_v2"
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
    
    # Daily Donchian channel (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Weekly trend filter: 20-week EMA on weekly closes
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20w = np.full(len(close_1w), np.nan)
    for i in range(19, len(close_1w)):
        if i == 19:
            ema_20w[i] = np.mean(close_1w[0:20])
        else:
            ema_20w[i] = close_1w[i] * 2/(20+1) + ema_20w[i-1] * (1 - 2/(20+1))
    ema_20w_aligned = align_htf_to_ltf(prices, df_1w, ema_20w)
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if weekly trend data not available
        if np.isnan(ema_20w_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
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
            # Look for entries with volume confirmation and weekly trend filter
            if volume_filter:
                # Breakout above Donchian high with weekly uptrend
                if (close[i] > donchian_high[i] and close[i-1] <= donchian_high[i] and 
                    close[i] > ema_20w_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Breakdown below Donchian low with weekly downtrend
                elif (close[i] < donchian_low[i] and close[i-1] >= donchian_low[i] and 
                      close[i] < ema_20w_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals