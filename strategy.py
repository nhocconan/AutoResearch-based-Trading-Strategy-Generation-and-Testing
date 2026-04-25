#!/usr/bin/env python3
"""
6h_ADX_DMI_Crossover_1dTrendFilter_VolumeConfirm
Hypothesis: 6h ADX(14) + DMI crossover with 1d EMA50 trend filter and volume confirmation.
ADX > 25 ensures trending markets; +DI crossing above -DI signals long, vice versa for short.
1d EMA50 filters for higher-timeframe trend alignment to avoid counter-trend whipsaws.
Volume spike (>1.8x 20-period average) confirms institutional participation.
Designed for 12-37 trades/year (50-150 over 4 years) on 6h timeframe to minimize fee drag.
Works in bull markets via trend continuation and bear markets via ADX regime filter preventing false signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for EMA50 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 trend filter (loaded ONCE)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h ADX(14) and DMI calculation
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    plus_di_6h = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_6h)
    minus_di_6h = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_6h)
    dx_6h = 100 * np.abs(plus_di_6h - minus_di_6h) / (plus_di_6h + minus_di_6h)
    adx_6h = pd.Series(dx_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Convert to arrays and handle NaN
    plus_di_6h_values = plus_di_6h.fillna(0).values
    minus_di_6h_values = minus_di_6h.fillna(0).values
    adx_6h_values = adx_6h.fillna(0).values
    
    # Volume spike: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for 1d EMA50 (50), 6h ADX/DMI (14+14), volume MA (20)
    start_idx = max(50, 28, 20)  # EMA50 + ADX(14+14) + vol MA20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(plus_di_6h_values[i]) or 
            np.isnan(minus_di_6h_values[i]) or np.isnan(adx_6h_values[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Look for entry signals - require: DMI crossover + ADX > 25 + 1d EMA50 trend alignment + volume spike
            bullish_crossover = plus_di_6h_values[i] > minus_di_6h_values[i] and plus_di_6h_values[i-1] <= minus_di_6h_values[i-1]
            bearish_crossover = minus_di_6h_values[i] > plus_di_6h_values[i] and minus_di_6h_values[i-1] <= plus_di_6h_values[i-1]
            
            # Trend filter: price must be on correct side of 1d EMA50
            long_trend = curr_close > ema_50_1d_aligned[i]
            short_trend = curr_close < ema_50_1d_aligned[i]
            
            # Regime filter: only trade when ADX > 25 (trending market)
            trending = adx_6h_values[i] > 25
            
            long_entry = (bullish_crossover and volume_spike[i] and long_trend and trending)
            short_entry = (bearish_crossover and volume_spike[i] and short_trend and trending)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when bearish crossover OR trend reverses OR ADX drops below 20 (losing momentum)
            bearish_crossover = minus_di_6h_values[i] > plus_di_6h_values[i] and minus_di_6h_values[i-1] <= plus_di_6h_values[i-1]
            if bearish_crossover or curr_close < ema_50_1d_aligned[i] or adx_6h_values[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when bullish crossover OR trend reverses OR ADX drops below 20 (losing momentum)
            bullish_crossover = plus_di_6h_values[i] > minus_di_6h_values[i] and plus_di_6h_values[i-1] <= minus_di_6h_values[i-1]
            if bullish_crossover or curr_close > ema_50_1d_aligned[i] or adx_6h_values[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ADX_DMI_Crossover_1dTrendFilter_VolumeConfirm"
timeframe = "6h"
leverage = 1.0