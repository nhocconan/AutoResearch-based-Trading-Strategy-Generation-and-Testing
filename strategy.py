#!/usr/bin/env python3
"""
4h_TrendBreakout_VolumeSpike_1dEMA34
Hypothesis: Trade breakouts from 4h Donchian channels (20) with 1d EMA34 trend filter and volume spike confirmation. Works in bull markets by catching breakouts and in bear markets by filtering trades to only those aligned with higher timeframe trend (1d EMA34). Volume spike ensures institutional participation. Targets 20-40 trades/year via strict breakout + volume + trend confluence. Uses ATR-based stoploss via signal=0 when price closes against position by 2x ATR.
"""

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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA34
    ema_period = 34
    ema_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period-1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 / (ema_period + 1)) + (ema_1d[i-1] * (1 - 2 / (ema_period + 1)))
    
    # Align 1d EMA34 to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 4h Donchian channel (20)
    lookback = 20
    upper = np.full_like(high, np.nan)
    lower = np.full_like(low, np.nan)
    
    for i in range(lookback-1, len(high)):
        upper[i] = np.max(high[i-lookback+1:i+1])
        lower[i] = np.min(low[i-lookback+1:i+1])
    
    # ATR for stoploss (20 period)
    atr_period = 20
    tr = np.zeros_like(high)
    atr = np.full_like(high, np.nan)
    
    for i in range(len(high)):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    if len(tr) >= atr_period:
        atr[atr_period-1] = np.mean(tr[:atr_period])
        for i in range(atr_period, len(tr)):
            atr[i] = (tr[i] + (atr_period-1) * atr[i-1]) / atr_period
    
    # Volume spike: volume > 2x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, vol_period, ema_period, atr_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        vol_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian + above 1d EMA34 + volume spike
            if close[i] > upper[i] and close[i] > ema_1d_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + below 1d EMA34 + volume spike
            elif close[i] < lower[i] and close[i] < ema_1d_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long: hold unless stoploss or reversal
            if close[i] < ema_1d_aligned[i] or close[i] < (high[i-max(1, lookback//2)] - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short: hold unless stoploss or reversal
            if close[i] > ema_1d_aligned[i] or close[i] > (low[i-max(1, lookback//2)] + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_TrendBreakout_VolumeSpike_1dEMA34"
timeframe = "4h"
leverage = 1.0