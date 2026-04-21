#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_ATRStop_V1
Hypothesis: 4h Donchian channel (20) breakout with volume confirmation (>2.0x 20-period volume MA) and ATR-based stoploss. Uses 1d HTF EMA50 for trend filter (price > EMA50 for longs, < EMA50 for shorts). Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drag. Works in bull markets via breakouts and in bear markets via short breakdowns. Volume spike ensures institutional participation. ATR stoploss adapts to volatility. Focus on BTC/ETH as primary, SOL as secondary.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA trend filter)
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
    
    # Donchian channel (20-period)
    highest_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss
    tr1 = pd.Series(high_4h - low_4h)
    tr2 = pd.Series(np.abs(high_4h - np.roll(close_4h, 1)))
    tr3 = pd.Series(np.abs(low_4h - np.roll(close_4h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i])
            or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = volume_4h[i]
        vol_ok = vol > 2.0 * vol_ma[i]  # volume confirmation (2x average)
        
        if position == 0:
            # Long: Donchian breakout above upper band + volume + trend filter (uptrend)
            if price > highest_high[i] and vol_ok and price > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Donchian breakdown below lower band + volume + trend filter (downtrend)
            elif price < lowest_low[i] and vol_ok and price < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit: price re-enters Donchian channel (failed breakout)
            elif price < highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit: price re-enters Donchian channel (failed breakdown)
            elif price > lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_ATRStop_V1"
timeframe = "4h"
leverage = 1.0