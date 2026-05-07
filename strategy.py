#!/usr/bin/env python3

# 4h_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike
# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike (>1.5x 20-period average).
# Designed for low-frequency, high-conviction trades in both bull and bear markets.
# Target: 20-40 trades/year per symbol to avoid fee drag.

name = "4h_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (R3, S3) using previous 4h candle
    camarilla_R3 = np.full(n, np.nan)
    camarilla_S3 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Previous period's OHLC
        ph = high[i-1]
        pl = low[i-1]
        pc = close[i-1]
        
        # Camarilla calculations
        range_val = ph - pl
        camarilla_R3[i] = pc + (range_val * 1.1000 / 4)
        camarilla_S3[i] = pc - (range_val * 1.1000 / 4)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = np.full_like(close_12h, np.nan)
    for i in range(50, len(close_12h)):
        ema_50_12h[i] = np.mean(close_12h[i-50:i])  # Simple MA for robustness
    
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 8  # Prevent overtrading (approx 4 days)
    
    start_idx = max(20, 50)  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_R3[i]) or np.isnan(camarilla_S3[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine 12h trend direction using EMA50
        trend_12h_up = close_12h[-1] > ema_50_12h[-1] if len(close_12h) > 0 else False  # Use last known value
        trend_12h_down = close_12h[-1] < ema_50_12h[-1] if len(close_12h) > 0 else False
        
        # More robust: use current aligned EMA vs price
        if not np.isnan(ema_50_12h_aligned[i]):
            # Need 12h close price aligned to 4h
            close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
            trend_12h_up = close_12h_aligned[i] > ema_50_12h_aligned[i]
            trend_12h_down = close_12h_aligned[i] < ema_50_12h_aligned[i]
        else:
            trend_12h_up = False
            trend_12h_down = False
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Camarilla R3 breakout in 12h uptrend with volume spike
            if (close[i] > camarilla_R3[i] and 
                trend_12h_up and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Camarilla S3 breakdown in 12h downtrend with volume spike
            elif (close[i] < camarilla_S3[i] and 
                  trend_12h_down and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit long: price crosses below camarilla S3 (or reverse signal)
            if close[i] < camarilla_S3[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above camarilla R3
            if close[i] > camarilla_R3[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals