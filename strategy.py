#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_12hEMA50_Trend_VolumeSpike_v4
Hypothesis: Trade 4h Camarilla R1/S1 breakouts aligned with 12h EMA50 trend and volume spikes (>2.0x 20-bar avg). Uses discrete sizing (0.30) to limit fee drag, with ATR-based stoploss. 12h EMA50 provides stronger trend filter than daily, reducing false breakouts in sideways markets. Designed for BTC/ETH resilience in both bull and bear regimes via tight entries (target: 25-40 trades/year) and volume confirmation.
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
    
    # Get 12h data for EMA50 trend filter and 1d for Camarilla
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 50 or len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla pivot levels from prior day's OHLC (1d)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Use prior day's OHLC (shift by 1 to avoid look-ahead)
    high_prev = np.roll(high_1d, 1)
    low_prev = np.roll(low_1d, 1)
    close_prev = np.roll(close_1d, 1)
    # For first bar, use first available
    high_prev[0] = high_1d[0]
    low_prev[0] = low_1d[0]
    close_prev[0] = close_1d[0]
    
    # Camarilla calculations
    pivot = (high_prev + low_prev + close_prev) / 3.0
    range_val = high_prev - low_prev
    r1 = close_prev + range_val * 1.1 / 12
    s1 = close_prev - range_val * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # ATR for stoploss calculation (4h ATR)
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # first bar
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of EMA50 (50), ATR (14), volume MA (20)
    start_idx = max(50, 14, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        ema_50_val = ema_50_12h_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        close_val = close[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R1, above 12h EMA50, with volume spike
            long_signal = (close_val > r1_val) and (close_val > ema_50_val) and vol_spike
            
            # Short: price breaks below S1, below 12h EMA50, with volume spike
            short_signal = (close_val < s1_val) and (close_val < ema_50_val) and vol_spike
            
            if long_signal:
                signals[i] = 0.30
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.30
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            # Exit: price breaks below S1 OR ATR stoploss (2.0*ATR below entry)
            if (close_val < s1_val) or (close_val < entry_price - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            # Exit: price breaks above R1 OR ATR stoploss (2.0*ATR above entry)
            if (close_val > r1_val) or (close_val > entry_price + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hEMA50_Trend_VolumeSpike_v4"
timeframe = "4h"
leverage = 1.0