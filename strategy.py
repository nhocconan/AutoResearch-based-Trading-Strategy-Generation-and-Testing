#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h EMA Pullback with 4h Trend and 1d Volume Filter
# Hypothesis: In trending markets, price pulls back to EMA before continuing.
# Use 4h EMA50 for trend direction, 1d volume spike for institutional interest.
# Enter on 1h EMA21 pullback in direction of 4h trend with volume confirmation.
# Works in both bull and bear by only trading with higher timeframe trend.
# Targets 15-35 trades/year via strict EMA pullback + volume + trend alignment.

name = "1h_ema_pullback_4h_trend_1d_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h EMA50 for trend filter (direction)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d volume average for spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    vol_avg_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # 1h EMA21 for pullback entries
    ema21 = pd.Series(close).ewm(span=21, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup for EMA and volume
        # Skip if required data not available
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i]) or 
            np.isnan(ema21[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x daily average volume
        vol_confirm = volume[i] > 2.0 * vol_avg_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below EMA21 OR trend changes
            if close[i] < ema21[i] or close[i] < ema50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short position
            # Exit: price closes above EMA21 OR trend changes
            if close[i] > ema21[i] or close[i] > ema50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long: pullback to EMA21 in uptrend with volume confirmation
            if (close[i] > ema21[i] and  # Price above EMA21 (not deep pullback)
                abs(close[i] - ema21[i]) < 0.5 * abs(ema21[i] - ema21[max(0, i-5)]) and  # Near EMA21
                close[i] > ema50_4h_aligned[i] and  # Uptrend on 4h
                vol_confirm):
                position = 1
                signals[i] = 0.20
            # Short: pullback to EMA21 in downtrend with volume confirmation
            elif (close[i] < ema21[i] and  # Price below EMA21 (not deep pullback)
                  abs(close[i] - ema21[i]) < 0.5 * abs(ema21[i] - ema21[max(0, i-5)]) and  # Near EMA21
                  close[i] < ema50_4h_aligned[i] and  # Downtrend on 4h
                  vol_confirm):
                position = -1
                signals[i] = -0.20
    
    return signals