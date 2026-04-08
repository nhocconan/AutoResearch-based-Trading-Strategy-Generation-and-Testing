#!/usr/bin/env python3
# 6h_cci_1d_ema_volume_v1
# Hypothesis: On 6h timeframe, use daily CCI(20) for mean reversion signals with EMA200 trend filter and volume confirmation.
# Long when CCI < -100 (oversold), price > EMA200, and volume > 1.5x average.
# Short when CCI > 100 (overbought), price < EMA200, and volume > 1.5x average.
# Exit when CCI crosses back above/below zero or volume drops below average.
# This strategy targets 50-150 total trades over 4 years by combining mean reversion with trend filter.
# Works in both bull and bear markets: mean reversion in ranges, trend filter avoids counter-trend trades in strong trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_cci_1d_ema_volume_v1"
timeframe = "6h"
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
    
    # Calculate daily CCI(20)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Typical Price
    tp = (daily_high + daily_low + daily_close) / 3
    # SMA of TP
    sma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    # Mean Deviation
    md = pd.Series(tp).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    # CCI
    cci = (tp - sma_tp) / (0.015 * md)
    # Handle division by zero or near-zero MD
    cci = np.where(md == 0, 0, cci)
    
    # Align daily CCI to 6h timeframe
    cci_6h = align_htf_to_ltf(prices, df_daily, cci)
    
    # Daily EMA200 trend filter
    ema200 = pd.Series(daily_close).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_6h = align_htf_to_ltf(prices, df_daily, ema200)
    
    # Volume confirmation: 20-period average on 6h
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 40  # Enough for CCI and EMA200 calculation
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(cci_6h[i]) or np.isnan(ema200_6h[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: CCI crosses above zero or volume drops below average
            if cci_6h[i] >= 0 or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI crosses below zero or volume drops below average
            if cci_6h[i] <= 0 or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Trend filter
            uptrend = close[i] > ema200_6h[i]
            downtrend = close[i] < ema200_6h[i]
            
            # Long entry: CCI oversold (< -100) with volume and uptrend
            if cci_6h[i] < -100 and volume_ok and uptrend:
                position = 1
                signals[i] = 0.25
            # Short entry: CCI overbought (> 100) with volume and downtrend
            elif cci_6h[i] > 100 and volume_ok and downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals