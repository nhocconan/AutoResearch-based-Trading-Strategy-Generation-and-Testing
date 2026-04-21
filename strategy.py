#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Trend
Hypothesis: Donchian(20) breakout on 4h timeframe with volume confirmation and trend filter (1d EMA200).
In uptrend (price > 1d EMA200), buy on breakout above Donchian high with volume > 1.5x 20-period average.
In downtrend (price < 1d EMA200), sell on breakout below Donchian low with volume > 1.5x 20-period average.
Exit when price crosses back through Donchian midpoint or trend reverses.
Designed for 4h timeframe to target 20-50 trades/year with high-conviction entries.
Works in bull markets by capturing continuation and in bear markets by capturing breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    ema = np.zeros_like(close)
    if len(close) >= period:
        ema[period-1] = np.mean(close[:period])
        multiplier = 2 / (period + 1)
        for i in range(period, len(close)):
            ema[i] = (close[i] - ema[i-1]) * multiplier + ema[i-1]
    return ema

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = np.zeros_like(tr)
    if len(tr) >= period:
        atr[period-1] = np.mean(tr[:period])
    
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily EMA200 for trend filter
    ema200_1d = calculate_ema(close_1d, 200)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(ema200_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only (avoid low-volume Asian session)
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        high = prices['high'].iloc[i]
        low = prices['low'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Donchian(20) channels
        if i >= 20:
            donch_high = prices['high'].iloc[i-20:i].max()
            donch_low = prices['low'].iloc[i-20:i].min()
            donch_mid = (donch_high + donch_low) / 2
        else:
            donch_high = prices['high'].iloc[:i+1].max()
            donch_low = prices['low'].iloc[:i+1].min()
            donch_mid = (donch_high + donch_low) / 2
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Uptrend: price > daily EMA200
            if price > ema200_1d_aligned[i]:
                # Long: price breaks above Donchian high with volume confirmation
                if (price > donch_high and volume_ok):
                    signals[i] = 0.30
                    position = 1
            # Downtrend: price < daily EMA200
            elif price < ema200_1d_aligned[i]:
                # Short: price breaks below Donchian low with volume confirmation
                if (price < donch_low and volume_ok):
                    signals[i] = -0.30
                    position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian midpoint or trend reverses
            if price < donch_mid or price < ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Short exit: price crosses above Donchian midpoint or trend reverses
            if price > donch_mid or price > ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0