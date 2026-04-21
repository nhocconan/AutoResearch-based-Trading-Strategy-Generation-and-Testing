#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout with 1d EMA50 trend filter and volume spike + ATR stoploss.
Breakout above/below Donchian channel (20-period high/low) with volume confirmation and aligned
trend captures momentum. EMA50 on 1d filters counter-trend trades. ATR-based stoploss limits
drawdown. Designed for 25-35 trades/year to minimize fee drag, works in bull/bear via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for trend filter and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d ATR for stoploss
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d_arr, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d_arr, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR is just high-low
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Load 4h data ONCE before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Donchian channels: 20-period high/low
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # Volume confirmation: volume / 20-period average volume (4h)
    vol_ma_20 = pd.Series(df_4h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_4h = df_4h['volume'].values / vol_ma_20
    vol_ratio_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(vol_ratio_aligned[i]) or
            np.isnan(atr_14_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_50_1d_aligned[i]
        upper = donch_high_aligned[i]
        lower = donch_low_aligned[i]
        vol_ratio = vol_ratio_aligned[i]
        atr = atr_14_aligned[i]
        vol_threshold = 1.5  # Volume must be 1.5x average
        
        if position == 0:
            # Enter long: price breaks above Donchian high, volume spike, uptrend
            if (price_close > upper and 
                vol_ratio > vol_threshold and 
                price_close > ema_trend):
                signals[i] = 0.30
                position = 1
                entry_price = price_close
                stop_price = entry_price - 2.5 * atr
            # Enter short: price breaks below Donchian low, volume spike, downtrend
            elif (price_close < lower and 
                  vol_ratio > vol_threshold and 
                  price_close < ema_trend):
                signals[i] = -0.30
                position = -1
                entry_price = price_close
                stop_price = entry_price + 2.5 * atr
        
        elif position != 0:
            # Check stoploss
            if position == 1 and price_close < stop_price:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > stop_price:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "4h_DonchianBreakout_1dEMA50_Trend_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0