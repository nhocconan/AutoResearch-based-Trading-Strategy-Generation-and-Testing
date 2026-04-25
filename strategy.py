#!/usr/bin/env python3
"""
1h_RSI_Regime_Donchian_Breakout_Volume
Hypothesis: On 1h timeframe, use 4h Donchian breakouts with volume confirmation and 1d RSI regime filter.
Only trade in trending regimes (RSI > 50 for longs, RSI < 50 for shorts) to avoid whipsaws in ranging markets.
Volume spike confirms breakout strength. Session filter (08-20 UTC) reduces noise.
Target: 15-30 trades/year by requiring multiple confluence factors.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h data for Donchian channels (loaded ONCE)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h Donchian channels (20-period)
    donchian_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_20)
    
    # 1d RSI for regime filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 1h volume spike filter (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for 4h Donchian (20) + 1d RSI (14) + 1h volume (20)
    start_idx = max(20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Regime filter: 1d RSI > 50 for uptrend, < 50 for downtrend
        uptrend_regime = rsi_1d_aligned[i] > 50
        downtrend_regime = rsi_1d_aligned[i] < 50
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        if position == 0:
            # Look for entry signals with regime and volume confirmation
            # Long breakout: price breaks above 4h Donchian high with uptrend regime + volume
            long_breakout = (curr_close > donchian_high_aligned[i]) and uptrend_regime and vol_confirm
            # Short breakout: price breaks below 4h Donchian low with downtrend regime + volume
            short_breakout = (curr_close < donchian_low_aligned[i]) and downtrend_regime and vol_confirm
            
            if long_breakout:
                signals[i] = 0.20
                position = 1
            elif short_breakout:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit if price breaks below Donchian low or regime changes
            if curr_close < donchian_low_aligned[i] or not uptrend_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position: exit if price breaks above Donchian high or regime changes
            if curr_close > donchian_high_aligned[i] or not downtrend_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI_Regime_Donchian_Breakout_Volume"
timeframe = "1h"
leverage = 1.0