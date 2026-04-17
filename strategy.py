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
    
    # === 12h ATR (14-period) for volatility filter ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR with Wilder's smoothing
    atr_12h = np.full_like(tr, np.nan)
    period = 14
    for i in range(len(tr)):
        if i < period:
            if i == 0:
                atr_12h[i] = tr[i]
            else:
                atr_12h[i] = (atr_12h[i-1] * (i-1) + tr[i]) / i
        else:
            atr_12h[i] = (atr_12h[i-1] * (period-1) + tr[i]) / period
    
    # === 12h EMA (34-period) for trend filter ===
    ema_34_12h = np.full_like(close_12h, np.nan)
    alpha = 2 / (34 + 1)
    for i in range(len(close_12h)):
        if np.isnan(ema_34_12h[i-1]) if i > 0 else True:
            ema_34_12h[i] = close_12h[i]
        else:
            ema_34_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema_34_12h[i-1]
    
    # === Align 12h indicators to 6h timeframe ===
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # === 6h Donchian Channel (20-period) for breakout signals ===
    # Highest high of last 20 periods
    highest_high = np.full_like(high, np.nan)
    for i in range(len(high)):
        if i >= 19:
            highest_high[i] = np.max(high[i-19:i+1])
        elif i > 0:
            highest_high[i] = np.max(high[max(0, i-9):i+1])
        else:
            highest_high[i] = high[0]
    
    # Lowest low of last 20 periods
    lowest_low = np.full_like(low, np.nan)
    for i in range(len(low)):
        if i >= 19:
            lowest_low[i] = np.min(low[i-19:i+1])
        elif i > 0:
            lowest_low[i] = np.min(low[max(0, i-9):i+1])
        else:
            lowest_low[i] = low[0]
    
    # === 6h Volume confirmation (20-period average) ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_confirm = volume > vol_ma_20 * 1.5
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_12h_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if volatility too low (ATR < 0.5% of price)
        if atr_12h_aligned[i] < close[i] * 0.005:
            signals[i] = 0.0
            position = 0
            continue
        
        # Trend filter: price above/below 12h EMA34
        uptrend = close[i] > ema_34_12h_aligned[i]
        downtrend = close[i] < ema_34_12h_aligned[i]
        
        # Entry logic
        if position == 0:
            # Long breakout: price breaks above Donchian high + uptrend + volume confirmation
            if (close[i] > highest_high[i] and 
                uptrend and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short breakout: price breaks below Donchian low + downtrend + volume confirmation
            elif (close[i] < lowest_low[i] and 
                  downtrend and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse signal or volatility spike
        elif position == 1:
            # Exit long: price breaks below Donchian low OR downtrend flip
            if (close[i] < lowest_low[i] or not uptrend):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high OR uptrend flip
            if (close[i] > highest_high[i] or not downtrend):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_DonchianBreakout_12hEMA_TrendFilter_VolumeConfirm"
timeframe = "6h"
leverage = 1.0