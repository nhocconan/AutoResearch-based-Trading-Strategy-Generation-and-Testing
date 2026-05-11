#!/usr/bin/env python3
"""
1h_Pullback_4DTrend_Volume
Hypothesis: In strong 4-day trends (1d EMA34), wait for pullbacks to the 21 EMA on 1h with volume confirmation.
Go long on bullish pullbacks in uptrend, short on bearish pullbacks in downtrend.
Uses 4h trend filter (EMA34) to avoid counter-trend trades and 1h for precise entry timing.
Designed for low trade frequency by requiring trend alignment, pullback to EMA, and volume spike.
Works in both bull and bear markets by following the 4-day trend.
"""
name = "1h_Pullback_4DTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # 1h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 4h Trend Filter (EMA34) ---
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # --- 1h EMA21 for pullback entries ---
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # --- Volume Spike Detection (20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(ema_21[i]) or
            np.isnan(vol_ratio[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: uptrend (price > 4h EMA34), pullback to 1h EMA21 with volume
            if (close[i] > ema_34_4h_aligned[i] and      # 4h uptrend
                close[i] <= ema_21[i] * 1.01 and         # near or slightly above 1h EMA21 (pullback)
                close[i] >= ema_21[i] * 0.99 and         # near or slightly below 1h EMA21
                volume_spike):
                signals[i] = 0.20
                position = 1
            # Short: downtrend (price < 4h EMA34), pullback to 1h EMA21 with volume
            elif (close[i] < ema_34_4h_aligned[i] and    # 4h downtrend
                  close[i] <= ema_21[i] * 1.01 and       # near or slightly above 1h EMA21 (pullback)
                  close[i] >= ema_21[i] * 0.99 and       # near or slightly below 1h EMA21
                  volume_spike):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions: trend reversal or loss of momentum
            if position == 1:
                # Exit long: 4h trend turns down or price breaks above EMA21 with momentum
                if (close[i] < ema_34_4h_aligned[i] * 0.995 or   # 4h trend broken
                    close[i] > ema_21[i] * 1.02):                # broken above pullback zone
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: 4h trend turns up or price breaks below EMA21 with momentum
                if (close[i] > ema_34_4h_aligned[i] * 1.005 or   # 4h trend broken
                    close[i] < ema_21[i] * 0.98):                # broken below pullback zone
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals