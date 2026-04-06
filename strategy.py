#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter and volume confirmation.
# Uses 4h Donchian channel breakouts for trend continuation.
# 12h trend filter (price above/below 12h EMA20) ensures alignment with higher timeframe trend.
# Volume confirmation (current volume > 1.5x 20-period average) filters low-quality breakouts.
# Works in bull markets via upward breakouts and in bear markets via downward breakdowns.
# Target: 75-200 trades over 4 years (19-50/year).

name = "4h_donchian20_12h_trend_vol_v1"
timeframe = "4h"
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
    
    # 4h Donchian channel (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # 12h trend filter: EMA20 on 12h closes
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_20_12h = np.full(len(close_12h), np.nan)
    for i in range(19, len(close_12h)):
        if i == 19:
            ema_20_12h[i] = np.mean(close_12h[0:20])
        else:
            ema_20_12h[i] = close_12h[i] * 2/(20+1) + ema_20_12h[i-1] * (1 - 2/(20+1))
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if trend or Donchian data not available
        if np.isnan(ema_20_12h_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]):
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
            # Look for entries with volume confirmation and 12h trend filter
            if volume_filter:
                # Breakout above Donchian high with 12h uptrend
                if (close[i] > donchian_high[i] and close[i-1] <= donchian_high[i] and 
                    close[i] > ema_20_12h_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Breakdown below Donchian low with 12h downtrend
                elif (close[i] < donchian_low[i] and close[i-1] >= donchian_low[i] and 
                      close[i] < ema_20_12h_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals