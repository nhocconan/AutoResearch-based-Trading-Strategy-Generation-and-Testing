#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeFilter_V2
Hypothesis: 4h Donchian(20) breakouts with 1d EMA50 trend filter and volume confirmation work on BTC and ETH in both bull and bear markets. Uses discrete position sizing (0.30) to limit fee drag and ATR-based stoploss for risk control. Target: 20-50 trades/year per symbol (80-200 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR for stoploss (using 15m data)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume filter (20-period average)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation
        volume_ok = volume > 1.5 * vol_ma[i]
        
        # 1d trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Calculate 20-period Donchian channels using prior 20 periods
            lookback_start = max(0, i - 20)
            lookback_end = i  # exclusive, so we use [lookback_start:lookback_end]
            if lookback_end - lookback_start >= 20:
                highest_high = np.max(high[lookback_start:lookback_end])
                lowest_low = np.min(low[lookback_start:lookback_end])
                
                # Long: price breaks above Donchian high in uptrend with volume
                if uptrend and volume_ok:
                    if price > highest_high:
                        signals[i] = 0.30
                        position = 1
                        entry_price = price
                # Short: price breaks below Donchian low in downtrend with volume
                elif downtrend and volume_ok:
                    if price < lowest_low:
                        signals[i] = -0.30
                        position = -1
                        entry_price = price
        
        elif position == 1:
            # Exit: price reaches Donchian low or stoploss
            lookback_start = max(0, i - 20)
            lookback_end = i
            if lookback_end - lookback_start >= 20:
                lowest_low = np.min(low[lookback_start:lookback_end])
                if price <= lowest_low or price < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.30
        
        elif position == -1:
            # Exit: price reaches Donchian high or stoploss
            lookback_start = max(0, i - 20)
            lookback_end = i
            if lookback_end - lookback_start >= 20:
                highest_high = np.max(high[lookback_start:lookback_end])
                if price >= highest_high or price > entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeFilter_V2"
timeframe = "4h"
leverage = 1.0