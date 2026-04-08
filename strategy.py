#!/usr/bin/env python3
"""
12h_1d_kama_rsi_volume_v1
Hypothesis: KAMA direction on 12h with RSI and volume confirmation for trend following.
- KAMA (Kaufman Adaptive Moving Average) adapts to market noise, reducing false signals in choppy markets.
- Entry: KAMA direction up/down + RSI > 50 (long) or < 50 (short) + volume > 1.5x 20-period average.
- Exit: KAMA direction reverses or RSI crosses back through 50.
- Position sizing: 0.25 long, -0.25 short.
- Designed to work in trending markets (KAMA follows trend) and avoid whipsaws in ranging markets (KAMA flattens).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_kama_rsi_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for KAMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d KAMA calculation
    close_1d = df_1d['close'].values
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder, will compute properly below
    # Recompute volatility correctly: sum of absolute changes over ER period
    er_period = 10
    change_abs = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility_sum = np.zeros_like(close_1d)
    for i in range(er_period, len(close_1d)):
        volatility_sum[i] = np.sum(np.abs(np.diff(close_1d[i-er_period:i+1])))
    er = np.zeros_like(close_1d)
    er[er_period:] = change_abs[er_period:] / np.where(volatility_sum[er_period:] == 0, 1, volatility_sum[er_period:])
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # KAMA direction: up if close > KAMA, down if close < KAMA
    kama_dir_up = close_1d > kama
    kama_dir_down = close_1d < kama
    
    # Forward fill direction
    kama_dir_up_series = pd.Series(kama_dir_up)
    kama_dir_down_series = pd.Series(kama_dir_down)
    kama_dir_up_ffilled = kama_dir_up_series.ffill().values
    kama_dir_down_ffilled = kama_dir_down_series.ffill().values
    
    # Align 1d KAMA direction to 12h
    kama_dir_up_aligned = align_htf_to_ltf(prices, df_1d, kama_dir_up_ffilled)
    kama_dir_down_aligned = align_htf_to_ltf(prices, df_1d, kama_dir_down_ffilled)
    
    # 12h RSI calculation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(kama_dir_up_aligned[i]) or np.isnan(kama_dir_down_aligned[i]) or
            np.isnan(rsi_values[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: KAMA direction down OR RSI < 50
            if kama_dir_down_aligned[i] or (rsi_values[i] < 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: KAMA direction up OR RSI > 50
            if kama_dir_up_aligned[i] or (rsi_values[i] > 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: KAMA up + RSI > 50 + volume
            if kama_dir_up_aligned[i] and (rsi_values[i] > 50) and volume_filter[i]:
                # Confirm KAMA just turned up (avoid whipsaw)
                if i > start_idx and not kama_dir_up_aligned[i-1]:
                    position = 1
                    signals[i] = 0.25
            # Short entry: KAMA down + RSI < 50 + volume
            elif kama_dir_down_aligned[i] and (rsi_values[i] < 50) and volume_filter[i]:
                # Confirm KAMA just turned down
                if i > start_idx and not kama_dir_down_aligned[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals