#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Uses discrete sizing 0.25 to limit fee drag. Target: 50-150 total trades over 4 years (12-37/year).
# Works in bull markets (breakouts with trend) and bear markets (breakouts against trend filtered out by 12h EMA50).
# Focus on BTC/ETH as primary symbols with proven edge from Donchian + volume + trend confluence.
# Weekly pivot direction from 1w timeframe adds institutional bias filter.

name = "6h_Donchian20_12hEMA50_VolumeSpike_1wPivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h Donchian(20) from prior 6h bar (breakout of prior 20-period channel)
    prior_20_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    prior_20_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 1w pivot direction (bullish if close > weekly pivot, bearish if < weekly pivot)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    # Weekly pivot = (weekly high + weekly low + weekly close) / 3
    weekly_pivot = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    pivot_values = weekly_pivot.values
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_values)
    
    # Volume confirmation: volume > 2.0x 20-period average (strict to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 20, 50, 20, 14)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(prior_20_high[i]) or np.isnan(prior_20_low[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donch_high = prior_20_high[i]
        curr_donch_low = prior_20_low[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_pivot = pivot_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_atr = atr_14[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with Donchian break and 12h EMA50 trend filter
            # Weekly pivot adds directional bias: long only above pivot, short only below pivot
            if curr_volume_spike:
                # Bullish: Close breaks above Donchian high + price above 12h EMA50 + above weekly pivot
                if curr_close > curr_donch_high and curr_close > curr_ema_50_12h and curr_close > curr_pivot:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish: Close breaks below Donchian low + price below 12h EMA50 + below weekly pivot
                elif curr_close < curr_donch_low and curr_close < curr_ema_50_12h and curr_close < curr_pivot:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2 * ATR below entry
            stop_loss = entry_price - 2.0 * curr_atr
            # Exit: Stoploss hit OR close drops below Donchian low OR loses 12h trend OR drops below weekly pivot
            if curr_low <= stop_loss or curr_close < curr_donch_low or curr_close < curr_ema_50_12h or curr_close < curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_loss = entry_price + 2.0 * curr_atr
            # Exit: Stoploss hit OR close rises above Donchian high OR loses 12h trend OR rises above weekly pivot
            if curr_high >= stop_loss or curr_close > curr_donch_high or curr_close > curr_ema_50_12h or curr_close > curr_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals