#!/usr/bin/env python3

# 12h_Camarilla_R3S3_Breakout_1dTrend_Volume_spike_v3
# Hypothesis: 12h breakout of Camarilla R3/S3 levels confirmed by 1d trend (EMA34) and volume spike (>2x 20-period average).
# Designed for low-frequency, high-conviction trades in both bull and bear markets.
# Target: 20-40 trades/year per symbol to avoid fee drag.

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume_spike_v3"
timeframe = "12h"
leverage = 1.0

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
    
    # Get 1d data for filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels (R3, S3, R2, S2) using previous 12h candle
    camarilla_R3 = np.full(n, np.nan)
    camarilla_S3 = np.full(n, np.nan)
    camarilla_R2 = np.full(n, np.nan)
    camarilla_S2 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Previous period's OHLC
        ph = high[i-1]
        pl = low[i-1]
        pc = close[i-1]
        
        # Camarilla calculations
        range_val = ph - pl
        camarilla_R3[i] = pc + (range_val * 1.1000 / 4)
        camarilla_S3[i] = pc - (range_val * 1.1000 / 4)
        camarilla_R2[i] = pc + (range_val * 1.1000 / 6)
        camarilla_S2[i] = pc - (range_val * 1.1000 / 6)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = (close_1d[i] * 2 / 35) + (ema_34_1d[i-1] * 33 / 35)
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = np.full_like(vol_1d, np.nan)
    for i in range(20, len(vol_1d)):
        vol_ma_20_1d[i] = np.mean(vol_1d[i-20:i])
    
    # Align 1d indicators to 12h
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Volume spike condition: current 1d volume > 2x 20-day average
    vol_spike = vol_1d > (2 * vol_ma_20_1d)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 8  # Prevent overtrading (approx 4 days)
    
    start_idx = max(20, 34)  # Warmup for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_R3[i]) or np.isnan(camarilla_S3[i]) or 
            np.isnan(camarilla_R2[i]) or np.isnan(camarilla_S2[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(close_1d_aligned[i]) or np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine 1d trend direction using EMA34
        trend_1d_up = close_1d_aligned[i] > ema_34_1d_aligned[i]
        trend_1d_down = close_1d_aligned[i] < ema_34_1d_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Camarilla R3 breakout in 1d uptrend with volume spike
            if (close[i] > camarilla_R3[i] and 
                trend_1d_up and 
                vol_spike_aligned[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Camarilla S3 breakdown in 1d downtrend with volume spike
            elif (close[i] < camarilla_S3[i] and 
                  trend_1d_down and 
                  vol_spike_aligned[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit long: price crosses below camarilla S2
            if close[i] < camarilla_S2[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above camarilla R2
            if close[i] > camarilla_R2[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals