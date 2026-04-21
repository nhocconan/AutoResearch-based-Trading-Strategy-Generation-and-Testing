#!/usr/bin/env python3
"""
4h_Donchian20_VolumeSpike_HTFTrend_ATRStop_V1
Hypothesis: 4h Donchian(20) breakout with 1d trend filter (price > 1d EMA50 for longs, < for shorts), volume confirmation (>1.5x 20-period volume MA), and ATR-based trailing stoploss. 
Donchian channels provide clear structure; EMA50 filters for higher-timeframe trend alignment; volume confirms breakout strength. 
ATR stoploss manages risk. Target 20-50 trades/year (75-200 total over 4 years).
Uses 4h primary timeframe with 1d HTF for EMA trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 4h Indicators (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss
    tr1 = pd.Series(high_4h - low_4h)
    tr2 = pd.Series(np.abs(high_4h - np.roll(close_4h, 1)))
    tr3 = pd.Series(np.abs(low_4h - np.roll(close_4h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    # Handle first value
    if np.isnan(atr[0]):
        atr[0] = tr.iloc[0] if len(tr) > 0 else 0.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) 
            or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = volume_4h[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        if position == 0:
            # Long: price breaks above Donchian high + volume confirmation + 1d uptrend
            if price > donchian_high[i] and vol_ok and price > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: price breaks below Donchian low + volume confirmation + 1d downtrend
            elif price < donchian_low[i] and vol_ok and price < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
        
        elif position == 1:
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, price)
            # ATR trailing stop: exit if price drops 2.5*ATR from high
            if price < highest_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit on Donchian low break (mean reversion)
            elif price < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, price)
            # ATR trailing stop: exit if price rises 2.5*ATR from low
            if price > lowest_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit on Donchian high break (mean reversion)
            elif price > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_HTFTrend_ATRStop_V1"
timeframe = "4h"
leverage = 1.0