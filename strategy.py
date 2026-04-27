# The strategy combines price action with volume and volatility filters to capture breakouts in trending markets while avoiding false signals in ranging conditions.
# It uses the Donchian channel for breakout detection, volume confirmation to validate breakout strength, and an ATR-based volatility filter to adapt to market conditions.
# The strategy is designed to work in both bull and bear markets by focusing on momentum breakouts with proper risk management.
# Timeframe: 4h (primary), HTF: 1d for trend context
# Entry conditions: Price breaks Donchian channel (20-period) with volume > 1.5x average and price above/below 1-day EMA50 for trend alignment
# Exit conditions: Opposite Donchian break or 2x ATR trailing stop
# Position sizing: Fixed at 0.25 (25% of capital) to balance risk and return
# Expected trade frequency: ~20-40 trades per year per symbol (within target range)

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close
    close_1d = df_1d['close'].values
    ema_50 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50[i] = (close_1d[i] * 2 + ema_50[i-1] * 48) / 50
    
    # Align EMA50 to 4h
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate ATR(14) for volatility filter
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate 20-period volume average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    # Calculate 20-period high/low for Donchian breakout
    high_max = np.full(n, np.nan)
    low_min = np.full(n, np.nan)
    period = 20
    for i in range(period, n):
        high_max[i] = np.max(high[i-period:i])
        low_min[i] = np.min(low[i-period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(14, vol_period, period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume AND above 1d EMA50
            if price > high_max[i] and vol_ratio > 1.5 and price > ema_50_aligned[i]:
                signals[i] = size
                position = 1
            # Short: Price breaks below Donchian low with volume AND below 1d EMA50
            elif price < low_min[i] and vol_ratio > 1.5 and price < ema_50_aligned[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below Donchian low or 2x ATR trailing stop
            if price < low_min[i] or price < high_max[i] - 2 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above Donchian high or 2x ATR trailing stop
            if price > high_max[i] or price > low_min[i] + 2 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_1dEMA50_Volume_Trend"
timeframe = "4h"
leverage = 1.0