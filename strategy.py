#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d trend filter and volume confirmation.
# Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures bull/bear strength.
# Trend filter: price above/below 1d EMA50 ensures alignment with daily trend.
# Volume confirmation: current volume > 1.3x 20-period average filters low-quality signals.
# Works in bull markets via Bull Power strength and in bear markets via Bear Power strength.
# Target: 80-200 trades over 4 years (20-50/year).

name = "6h_elder_ray_trend_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray: EMA13 of close
    ema13 = np.full(n, np.nan)
    for i in range(12, n):
        if i == 12:
            ema13[i] = np.mean(close[0:13])
        else:
            ema13[i] = close[i] * 2/(13+1) + ema13[i-1] * (1 - 2/(13+1))
    
    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # 1d trend filter: EMA50 on daily closes
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50d = np.full(len(close_1d), np.nan)
    for i in range(49, len(close_1d)):
        if i == 49:
            ema_50d[i] = np.mean(close_1d[0:50])
        else:
            ema_50d[i] = close_1d[i] * 2/(50+1) + ema_50d[i-1] * (1 - 2/(50+1))
    ema_50d_aligned = align_htf_to_ltf(prices, df_1d, ema_50d)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(ema13[i]) or np.isnan(ema_50d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Bear Power turns negative or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.5 * atr_approx
            
            if (bear_power[i] < 0 or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Bull Power turns negative or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.5 * atr_approx
            
            if (bull_power[i] < 0 or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend filter
            if volume_filter:
                # Long: Bull Power positive and price above 1d EMA50
                if (bull_power[i] > 0 and 
                    close[i] > ema_50d_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: Bear Power positive and price below 1d EMA50
                elif (bear_power[i] > 0 and 
                      close[i] < ema_50d_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals