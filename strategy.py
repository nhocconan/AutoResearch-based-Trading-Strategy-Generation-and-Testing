#!/usr/bin/env python3
"""
4h_BollingerBreakout_1dTrend_VolumeSpike_RSIFilter
Hypothesis: Bollinger Band breakouts filtered by daily trend and volume spikes with RSI confirmation.
Works in bull markets by catching breakouts above upper BB, and in bear markets by catching breakdowns below lower BB.
Daily EMA50 provides trend filter, volume spike confirms breakout strength, RSI prevents overextended entries.
Designed for low trade frequency (20-40/year) to minimize fee drag.
"""

name = "4h_BollingerBreakout_1dTrend_VolumeSpike_RSIFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2.0
    sma = np.full(n, np.nan)
    if n >= bb_period:
        sma[bb_period-1] = np.mean(close[0:bb_period])
        for i in range(bb_period, n):
            sma[i] = (sma[i-1] * (bb_period-1) + close[i]) / bb_period
    
    var = np.full(n, np.nan)
    if n >= bb_period:
        for i in range(bb_period-1, n):
            if i == bb_period-1:
                var[i] = np.var(close[0:bb_period])
            else:
                var[i] = (var[i-1] * (bb_period-1) + (close[i] - sma[i])**2) / bb_period
    
    std_dev = np.sqrt(var)
    upper_band = sma + bb_std * std_dev
    lower_band = sma - bb_std * std_dev
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (ema_50_1d[i-1] * 49 + close_1d[i]) / 50
    
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    # RSI (14) for momentum confirmation
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    if n >= rsi_period:
        avg_gain[rsi_period-1] = np.mean(gain[0:rsi_period])
        avg_loss[rsi_period-1] = np.mean(loss[0:rsi_period])
        for i in range(rsi_period, n):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.full(n, np.nan)
    rsi = np.full(n, np.nan)
    valid_rsi = (~np.isnan(avg_gain)) & (~np.isnan(avg_loss)) & (avg_loss != 0)
    rs[valid_rsi] = avg_gain[valid_rsi] / avg_loss[valid_rsi]
    rsi[valid_rsi] = 100 - (100 / (1 + rs[valid_rsi]))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(bb_period, 20, rsi_period)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ratio[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Enter long: price breaks above upper BB AND uptrend (price > EMA50) AND volume spike AND RSI not overbought
            if (close[i] > upper_band[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_ratio[i] > 2.0 and 
                rsi[i] < 70):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Enter short: price breaks below lower BB AND downtrend (price < EMA50) AND volume spike AND RSI not oversold
            elif (close[i] < lower_band[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_ratio[i] > 2.0 and 
                  rsi[i] > 30):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Minimum holding period: 3 bars
            if bars_since_entry < 3:
                signals[i] = 0.25
            else:
                # Exit long: price breaks below middle band OR trend reversal (price < EMA50)
                if close[i] < sma[i] or close[i] < ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Minimum holding period: 3 bars
            if bars_since_entry < 3:
                signals[i] = -0.25
            else:
                # Exit short: price breaks above middle band OR trend reversal (price > EMA50)
                if close[i] > sma[i] or close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals