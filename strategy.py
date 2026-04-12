#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter + volume confirmation + ATR stoploss
    # Uses 1d EMA50 for trend: only take breakouts in direction of daily trend
    # Volume: > 2.0 * 20-period average to filter false breakouts
    # Stoploss: ATR(14) * 2.0 from entry price (implemented as signal->0 when stop hit)
    # Discrete sizing 0.25 to minimize fee churn. Target: 20-40 trades/year per symbol.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate ATR(14) for stoploss
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(np.abs(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])  # first tr is NaN
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.nanmean(tr[i-13:i+1])  # ATR(14)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = np.zeros(n)  # track entry price for stoploss
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            entry_price[i] = 0.0
            continue
        
        # Determine 1d trend
        bullish_trend = close[i] > ema50_1d_aligned[i]
        bearish_trend = close[i] < ema50_1d_aligned[i]
        
        # Entry logic: Donchian breakout with volume and trend filter
        long_entry = False
        short_entry = False
        
        # Long breakout: price breaks above Donchian high in bullish trend
        if bullish_trend:
            long_entry = (close[i] > donchian_high[i]) and volume_spike[i]
        # Short breakout: price breaks below Donchian low in bearish trend
        elif bearish_trend:
            short_entry = (close[i] < donchian_low[i]) and volume_spike[i]
        
        # Stoploss logic: ATR(14) * 2.0 from entry price
        long_stop = False
        short_stop = False
        if position == 1 and i > 0:
            long_stop = close[i] < (entry_price[i-1] - 2.0 * atr[i])
        if position == -1 and i > 0:
            short_stop = close[i] > (entry_price[i-1] + 2.0 * atr[i])
        
        # Exit logic: opposite Donchian level
        long_exit = (not bullish_trend) or (close[i] < donchian_low[i])
        short_exit = (not bearish_trend) or (close[i] > donchian_high[i])
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
            entry_price[i] = close[i]
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
            entry_price[i] = close[i]
        elif position == 1 and (long_exit or long_stop):
            position = 0
            signals[i] = 0.0
            entry_price[i] = 0.0
        elif position == -1 and (short_exit or short_stop):
            position = 0
            signals[i] = 0.0
            entry_price[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
                entry_price[i] = entry_price[i-1] if i > 0 else close[i]
            elif position == -1:
                signals[i] = -0.25
                entry_price[i] = entry_price[i-1] if i > 0 else close[i]
            else:
                signals[i] = 0.0
                entry_price[i] = 0.0
    
    return signals

name = "4h_1d_donchian_breakout_trend_volume_atrstop_v1"
timeframe = "4h"
leverage = 1.0