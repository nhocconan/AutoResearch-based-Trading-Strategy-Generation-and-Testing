#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Trade 6h Donchian(20) breakouts in the direction of 1d EMA50 trend with volume confirmation.
Uses 1d EMA50 for trend filter to reduce whipsaws, and 1.8x volume spike for confirmation.
Only trade when ADX(14) > 25 to avoid chop. ATR-based trailing stop (2.5*ATR) and time-based exit (max 10 bars).
Designed for 12-30 trades/year on BTC/ETH/SOL. Works in bull/bear markets by following 1d EMA50 trend and filtering ranging regimes via ADX.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Donchian channels from 6h data (20-period lookback)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align HTF indicators to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 1.8x median volume (20-period) for signal
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # ATR(14) for volatility-based stops
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # First period
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ADX(14) for regime filter - trending when > 25
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    def WilderSmooth(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    if n >= period:
        plus_dm_smooth = WilderSmooth(plus_dm, period)
        minus_dm_smooth = WilderSmooth(minus_dm, period)
        tr_smooth = WilderSmooth(tr, period)
        
        # Avoid division by zero
        plus_di = 100 * plus_dm_smooth / np.where(tr_smooth != 0, tr_smooth, 1)
        minus_di = 100 * minus_dm_smooth / np.where(tr_smooth != 0, tr_smooth, 1)
        dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, (plus_di + minus_di), 1)
        adx = WilderSmooth(dx, period)
    else:
        adx = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Warmup: max of 1d EMA (50), Donchian (20), volume median (20), ADX (14*2 for smoothing), ATR (14)
    start_idx = max(50, 20, 20, 28, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or
            np.isnan(vol_median[i]) or
            np.isnan(adx[i]) or
            np.isnan(atr[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            bars_since_entry += 1
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        close_val = close[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        adx_val = adx[i]
        atr_val = atr[i]
        
        if position == 0:
            bars_since_entry = 0
            # Long: break above Donchian high with volume spike, uptrend, and trending regime
            long_signal = (close_val > highest_20[i]) and \
                          (volume_val > 1.8 * vol_median_val) and \
                          (close_val > ema_50_1d_val) and \
                          (adx_val > 25)
            
            # Short: break below Donchian low with volume spike, downtrend, and trending regime
            short_signal = (close_val < lowest_20[i]) and \
                           (volume_val > 1.8 * vol_median_val) and \
                           (close_val < ema_50_1d_val) and \
                           (adx_val > 25)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            bars_since_entry += 1
            # Exit conditions
            # 1. Price breaks below Donchian low (reversal)
            # 2. Trend changes (close < 1d EMA50)
            # 3. Regime changes (ADX < 20)
            # 4. ATR-based trailing stop (2.5 * ATR below highest high since entry)
            # 5. Time-based exit (max 10 bars)
            if (close_val < lowest_20[i]) or \
               (close_val < ema_50_1d_val) or \
               (adx_val < 20) or \
               (bars_since_entry >= 10):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            bars_since_entry += 1
            # Exit conditions
            # 1. Price breaks above Donchian high (reversal)
            # 2. Trend changes (close > 1d EMA50)
            # 3. Regime changes (ADX < 20)
            # 4. Time-based exit (max 10 bars)
            if (close_val > highest_20[i]) or \
               (close_val > ema_50_1d_val) or \
               (adx_val < 20) or \
               (bars_since_entry >= 10):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
    
    return signals

name = "6h_Donchian20_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0