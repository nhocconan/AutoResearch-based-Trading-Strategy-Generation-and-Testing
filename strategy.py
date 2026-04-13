#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA trend filter and volume confirmation.
# Elder Ray measures bull/bear power relative to EMA13. In strong trends, power expands; in ranges, it contracts.
# Combined with 1d EMA50 trend filter and volume spikes (2x avg volume), it filters false signals.
# Target: 15-30 trades per year (60-120 total over 4 years) for 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # EMA(50) for 1d trend filter
    ema50_1d = np.zeros(len(close_1d))
    ema_multiplier = 2 / (50 + 1)
    ema50_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema50_1d[i] = (close_1d[i] - ema50_1d[i-1]) * ema_multiplier + ema50_1d[i-1]
    
    # Align 1d EMA to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Elder Ray on 6h timeframe: Bull Power = High - EMA13, Bear Power = Low - EMA13
    # EMA13 for 6h
    ema13 = np.zeros(n)
    ema_multiplier_13 = 2 / (13 + 1)
    ema13[0] = close[0]
    for i in range(1, n):
        ema13[i] = (close[i] - ema13[i-1]) * ema_multiplier_13 + ema13[i-1]
    
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Average volume (20-period = 20*6h = 5 days) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_trend = ema50_1d_aligned[i]
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirm = vol > 2.0 * avg_vol
        
        if position == 0:
            # Long: Strong bull power (High > EMA13) + above 1d EMA50 + volume confirmation
            if (bull_power[i] > 0 and
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Strong bear power (Low < EMA13) + below 1d EMA50 + volume confirmation
            elif (bear_power[i] < 0 and
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bull power weakens (< 0) or price breaks below 1d EMA
            if (bull_power[i] <= 0 or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Bear power weakens (>= 0) or price breaks above 1d EMA
            if (bear_power[i] >= 0 or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_ElderRay_Trend_Volume"
timeframe = "6h"
leverage = 1.0