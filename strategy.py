#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter and volume confirmation.
# Uses 4h Donchian breakouts for entry. 1d EMA(50) filters direction: long only when 4h close > 1d EMA(50), short only when 4h close < 1d EMA(50).
# Volume confirmation: current volume > 1.8x 20-period average filters low-quality breakouts.
# Stoploss: 2.5x ATR(14) from entry. Designed to work in both bull and bear markets by following the 1d trend.
# Target: 75-200 trades over 4 years (19-50/year).

name = "4h_donchian20_1d_ema_vol_v2"
timeframe = "4h"
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
    
    # 4h Donchian channel (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50d = np.full(len(close_1d), np.nan)
    for i in range(len(close_1d)):
        if i < 49:
            ema_50d[i] = np.nan
        elif i == 49:
            ema_50d[i] = np.mean(close_1d[0:50])
        else:
            ema_50d[i] = close_1d[i] * 2/(50+1) + ema_50d[i-1] * (1 - 2/(50+1))
    ema_50d_aligned = align_htf_to_ltf(prices, df_1d, ema_50d)
    
    # Volume filter: current volume > 1.8x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr = np.full(n, np.nan)
    
    # Calculate ATR(14)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema_50d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.8
        
        # Check exits and stoploss
        if position == 1:  # long position
            stop_loss_level = entry_price - 2.5 * atr[i]
            if close[i] < donchian_low[i] or close[i] < stop_loss_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            stop_loss_level = entry_price + 2.5 * atr[i]
            if close[i] > donchian_high[i] or close[i] > stop_loss_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and 1d trend filter
            if volume_filter:
                # Breakout above Donchian high with 1d uptrend
                if close[i] > donchian_high[i] and close[i-1] <= donchian_high[i] and close[i] > ema_50d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Breakdown below Donchian low with 1d downtrend
                elif close[i] < donchian_low[i] and close[i-1] >= donchian_low[i] and close[i] < ema_50d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals