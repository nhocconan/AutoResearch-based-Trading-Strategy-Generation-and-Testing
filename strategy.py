#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeATR
Hypothesis: Donchian channel breakout on 4h with 1d EMA50 trend filter and volume/ATR confirmation.
Only trade breakouts in direction of daily trend when volume > 1.5x average and ATR > 0.5% of price.
Designed for low trade frequency (~20-30/year) to work in both bull and bear markets via trend alignment.
Donchian breakouts capture strong momentum with clear exit when price reverts to midpoint.
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF EMA50 to 4h timeframe (standard 1-bar delay for EMA)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d, additional_delay_bars=1)
    
    # Calculate 20-period Donchian channels on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Calculate volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    # Calculate ATR(14) for volatility filter and stoploss
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR filter: only trade when ATR > 0.5% of price (ensures sufficient volatility)
    atr_filter = atr > (0.005 * close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian (20), EMA50 (50), ATR (14), volume MA (20)
    start_idx = max(20, 50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for Donchian breakout signals with trend and confirmation filters
            # Long: price breaks above upper Donchian in uptrend with volume/ATR confirmation
            # Short: price breaks below lower Donchian in downtrend with volume/ATR confirmation
            long_signal = (close[i] > highest_high[i]) and (close[i] > ema50_aligned[i]) and volume_confirm[i] and atr_filter[i]
            short_signal = (close[i] < lowest_low[i]) and (close[i] < ema50_aligned[i]) and volume_confirm[i] and atr_filter[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price reverts to Donchian midpoint (mean reversion within channel)
            exit_signal = close[i] < donchian_mid[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price reverts to Donchian midpoint (mean reversion within channel)
            exit_signal = close[i] > donchian_mid[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeATR"
timeframe = "4h"
leverage = 1.0