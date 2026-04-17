#!/usr/bin/env python3
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
    
    # === 1w High-Low Range (weekly range) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    weekly_range = high_1w - low_1w
    
    # === 1d EMA(34) for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34 = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i == 0:
            ema_34[i] = close_1d[i]
        else:
            ema_34[i] = (close_1d[i] * 2 / (34 + 1)) + (ema_34[i-1] * (33 / (34 + 1)))
    
    # === 6h Volume Spike Detector ===
    vol_ma_20 = np.full_like(volume, np.nan)
    vol_std_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
            vol_std_20[i] = np.std(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
            vol_std_20[i] = np.std(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[i]
            vol_std_20[i] = 0.0
    
    volume_spike = volume > (vol_ma_20 + 2.0 * vol_std_20)
    
    # === Align HTF indicators to 6h timeframe ===
    weekly_range_aligned = align_htf_to_ltf(prices, df_1w, weekly_range)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # === 6h EMA(8) and EMA(21) for momentum ===
    ema_8 = np.full_like(close, np.nan)
    ema_21 = np.full_like(close, np.nan)
    for i in range(len(close)):
        if i == 0:
            ema_8[i] = close[i]
            ema_21[i] = close[i]
        else:
            ema_8[i] = (close[i] * 2 / (8 + 1)) + (ema_8[i-1] * (7 / (8 + 1)))
            ema_21[i] = (close[i] * 2 / (21 + 1)) + (ema_21[i-1] * (20 / (21 + 1)))
    
    # === 6h RSI(14) for overbought/oversold ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    period = 14
    for i in range(len(gain)):
        if i < period:
            if i == 0:
                avg_gain[i] = gain[i]
                avg_loss[i] = loss[i]
            else:
                avg_gain[i] = (avg_gain[i-1] * (i-1) + gain[i]) / i
                avg_loss[i] = (avg_loss[i-1] * (i-1) + loss[i]) / i
        else:
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss == 0] = 100
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_range_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(ema_8[i]) or 
            np.isnan(ema_21[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long conditions:
            # 1. Price above weekly range midpoint (bullish bias)
            # 2. EMA8 > EMA21 (short-term momentum up)
            # 3. RSI < 40 (not overbought, room to rise)
            # 4. Volume spike (participation)
            weekly_midpoint = (high_1w[i//28] + low_1w[i//28]) / 2 if i >= 28 else close[i]  # approximate weekly index
            if (close[i] > weekly_midpoint and 
                ema_8[i] > ema_21[i] and 
                rsi[i] < 40 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short conditions:
            # 1. Price below weekly range midpoint (bearish bias)
            # 2. EMA8 < EMA21 (short-term momentum down)
            # 3. RSI > 60 (not oversold, room to fall)
            # 4. Volume spike (participation)
            elif (close[i] < weekly_midpoint and 
                  ema_8[i] < ema_21[i] and 
                  rsi[i] > 60 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: EMA8 crosses below EMA21 OR RSI > 70 (overbought)
            if ema_8[i] < ema_21[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: EMA8 crosses above EMA21 OR RSI < 30 (oversold)
            if ema_8[i] > ema_21[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_EMA_RSI_VolumeSpike_WeeklyBias_v1"
timeframe = "6h"
leverage = 1.0