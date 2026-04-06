#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(10) breakout with weekly EMA(34) trend filter and volume confirmation.
# Shorter Donchian period for more signals, weekly EMA for trend filter, volume to filter false breakouts.
# Designed for 50-150 trades over 4 years (12-38/year) to balance opportunity and fee drag.
# Works in bull/bear by only trading with higher timeframe trend.

name = "1d_donchian10_weekly_ema34_vol_v1"
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
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # 34-period EMA on weekly timeframe (slower trend filter)
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    ema_weekly = np.full(len(close_weekly), np.nan)
    if len(close_weekly) >= 34:
        ema_weekly[33] = np.mean(close_weekly[:34])
        for i in range(34, len(close_weekly)):
            ema_weekly[i] = (close_weekly[i] * 2 + ema_weekly[i-1] * 32) / 34
    
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # 10-period Donchian channels on daily (shorter for more signals)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(10, n):
        donch_high[i] = np.max(high[i-10:i])
        donch_low[i] = np.min(low[i-10:i])
    
    # Volume filter: current volume > 1.3x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 34, 20, 10)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(ema_weekly_aligned[i]) or 
            np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
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
            # Exit: price closes below weekly EMA or stoploss hit
            if (close[i] < ema_weekly_aligned[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above weekly EMA or stoploss hit
            if (close[i] > ema_weekly_aligned[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: price breaks above Donchian high with volume and above weekly EMA (bullish)
            if (close[i] > donch_high[i] and volume_filter and 
                close[i] > ema_weekly_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low with volume and below weekly EMA (bearish)
            elif (close[i] < donch_low[i] and volume_filter and 
                  close[i] < ema_weekly_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals