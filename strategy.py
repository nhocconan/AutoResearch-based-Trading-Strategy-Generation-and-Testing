#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Donchian(20) breakout with daily EMA(50) trend filter and volume confirmation.
# Uses weekly trend to avoid counter-trend trades, volume to filter false breakouts.
# Targets 10-20 trades/year (40-80 over 4 years) to minimize fee drag.
# Works in bull/bear by only trading with higher timeframe trend.

name = "1d_weekly_donchian20_ema50_vol_v1"
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
    
    # 50-period EMA on weekly timeframe
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    ema_weekly = np.full(len(close_weekly), np.nan)
    if len(close_weekly) >= 50:
        ema_weekly[49] = np.mean(close_weekly[:50])
        for i in range(50, len(close_weekly)):
            ema_weekly[i] = (close_weekly[i] * 2 + ema_weekly[i-1] * 48) / 50
    
    ema_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # 20-period Donchian channels on weekly
    donch_high = np.full(len(close_weekly), np.nan)
    donch_low = np.full(len(close_weekly), np.nan)
    for i in range(20, len(close_weekly)):
        donch_high[i] = np.max(close_weekly[i-20:i])
        donch_low[i] = np.min(close_weekly[i-20:i])
    
    donch_high_aligned = align_htf_to_ltf(prices, df_weekly, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_weekly, donch_low)
    
    # Volume filter: current volume > 1.5x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(30, 50, 20, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(ema_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below weekly EMA or stoploss hit
            if (close[i] < ema_aligned[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above weekly EMA or stoploss hit
            if (close[i] > ema_aligned[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: price closes above weekly Donchian high with volume and above weekly EMA (bullish)
            if (close[i] > donch_high_aligned[i] and volume_filter and 
                close[i] > ema_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price closes below weekly Donchian low with volume and below weekly EMA (bearish)
            elif (close[i] < donch_low_aligned[i] and volume_filter and 
                  close[i] < ema_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals