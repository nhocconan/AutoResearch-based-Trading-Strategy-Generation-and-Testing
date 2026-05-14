#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation
# Donchian breakouts capture momentum moves. Weekly pivot (from 1d data) provides structural bias:
# - Price above weekly pivot (PP) = bullish bias, look for longs on breakouts
# - Price below weekly pivot = bearish bias, look for shorts on breakdowns
# Volume spike (>1.8x 20-bar average) confirms breakout validity and filters false signals.
# Works in bull/bear markets by following the higher-timeframe structural bias while using
# Donchian breakouts for precise entry timing. Target 12-37 trades/year to minimize fee drag.

name = "6h_Donchian20_WeeklyPivot_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points from daily OHLC (using prior week's data)
    # We need to group daily data into weeks and calculate pivot for each week
    # For simplicity, we'll use rolling weekly lookback: highest high, lowest low, close of prior 5 trading days
    df_1d = df_1d.copy()
    df_1d['weekly_high'] = df_1d['high'].rolling(window=5, min_periods=5).max().shift(1)  # prior week high
    df_1d['weekly_low'] = df_1d['low'].rolling(window=5, min_periods=5).min().shift(1)    # prior week low
    df_1d['weekly_close'] = df_1d['close'].rolling(window=5, min_periods=5).mean().shift(1)  # prior week close
    
    # Weekly pivot point: (H + L + C) / 3
    weekly_pp = (df_1d['weekly_high'] + df_1d['weekly_low'] + df_1d['weekly_close']) / 3.0
    weekly_pp_vals = weekly_pp.values
    
    # Align weekly pivot to 6h timeframe (completed weekly pivot only)
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1d, weekly_pp_vals)
    
    # Donchian channel (20-period) on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 20)  # Donchian20, volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_pp_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        weekly_pp_val = weekly_pp_aligned[i]
        upper_channel = donchian_high[i]
        lower_channel = donchian_low[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper channel AND price > weekly pivot (bullish bias) AND volume spike
            if price > upper_channel and price > weekly_pp_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price breaks below Donchian lower channel AND price < weekly pivot (bearish bias) AND volume spike
            elif price < lower_channel and price < weekly_pp_val and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on stoploss or breakdown below lower channel
            # ATR-based stoploss: 2.5 * ATR below entry (using 6h ATR)
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price - 2.5 * atr_val
            # Exit on stoploss or price breaks below Donchian lower channel (trend reversal)
            if price < stop_loss or price < lower_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on stoploss or breakout above upper channel
            # ATR-based stoploss: 2.5 * ATR above entry
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price + 2.5 * atr_val
            # Exit on stoploss or price breaks above Donchian upper channel (trend reversal)
            if price > stop_loss or price > upper_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals