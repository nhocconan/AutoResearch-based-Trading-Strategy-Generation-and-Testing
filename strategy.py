#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d trend filter (EMA50) and volume confirmation
# Donchian breakouts capture strong momentum moves. 1d EMA50 ensures alignment with daily trend.
# Volume confirmation filters false breakouts. Designed for 6h timeframe targeting 12-37 trades/year.
# Works in bull markets via long breakouts above 20-period high + uptrend filter.
# Works in bear markets via short breakouts below 20-period low + downtrend filter.

name = "6h_Donchian20_Breakout_1dEMA50_Trend_Volume_v1"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align 1d EMA to 6h (changes only when 1d bar closes)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 6h data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 50, 14)  # Donchian(20), 1d EMA(50), ATR(14)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(volume_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Price > Donchian high(20), above 1d EMA50, volume spike
            if price > highest_20[i] and price > ema_50_1d_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: Price < Donchian low(20), below 1d EMA50, volume spike
            elif price < lowest_20[i] and price < ema_50_1d_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on stoploss or retracement to 1d EMA50
            # ATR-based stoploss: 2.0 * ATR below entry
            stop_loss = entry_price - 2.0 * atr[i]
            if price < stop_loss or price < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on stoploss or retracement to 1d EMA50
            # ATR-based stoploss: 2.0 * ATR above entry
            stop_loss = entry_price + 2.0 * atr[i]
            if price > stop_loss or price > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals