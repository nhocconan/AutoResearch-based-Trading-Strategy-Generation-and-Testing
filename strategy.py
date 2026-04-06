#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_donchian20_ema50_vol_v4"
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
    
    # Get weekly data for trend filter and Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Calculate EMA(50) on weekly
    def ema(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        alpha = 2.0 / (period + 1)
        ema_val = np.full_like(arr, np.nan)
        ema_val[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            ema_val[i] = alpha * arr[i] + (1 - alpha) * ema_val[i-1]
        return ema_val
    
    ema_50_weekly = ema(close_weekly, 50)
    ema_50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)
    
    # Weekly Donchian channels (20-period)
    donchian_high_weekly = np.full_like(close_weekly, np.nan)
    donchian_low_weekly = np.full_like(close_weekly, np.nan)
    for i in range(20, len(close_weekly)):
        donchian_high_weekly[i] = np.max(high_weekly[i-20:i])
        donchian_low_weekly[i] = np.min(low_weekly[i-20:i])
    
    donchian_high_weekly_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high_weekly)
    donchian_low_weekly_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low_weekly)
    
    # Volume filter: current volume > 1.8x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(ema_50_weekly_aligned[i]) or \
           np.isnan(donchian_high_weekly_aligned[i]) or np.isnan(donchian_low_weekly_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.8
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below weekly Donchian low or stoploss hit
            if (close[i] < donchian_low_weekly_aligned[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above weekly Donchian high or stoploss hit
            if (close[i] > donchian_high_weekly_aligned[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries - only long in bull, only short in bear based on weekly trend
            # Long: price breaks above weekly Donchian high, above weekly EMA50, with volume (only in bull market)
            if (close[i] > donchian_high_weekly_aligned[i] and 
                close[i] > ema_50_weekly_aligned[i] and 
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below weekly Donchian low, below weekly EMA50, with volume (only in bear market)
            elif (close[i] < donchian_low_weekly_aligned[i] and 
                  close[i] < ema_50_weekly_aligned[i] and 
                  volume_filter):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals