#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Exponential Moving Average (EMA) Crossover with 1d Volume Confirmation and 1w Trend Filter
# - EMA(9) and EMA(21) crossover for trend momentum
# - 1d volume > 1.5x 20-period average for confirmation
# - 1w EMA(50) trend filter: only take longs when price > weekly EMA50, shorts when price < weekly EMA50
# - Designed to capture trending moves with volume confirmation while avoiding choppy markets
# - Target: 20-40 trades/year to minimize fee drag

name = "4h_EMACross_1dVolume_1wTrend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA(50) for trend direction
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # EMA crossover: EMA(9) and EMA(21)
    ema_9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(ema_9[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 4h volume > 1.5x 1d average volume (scaled)
        # Scale 1d average to 4h: 1d has 6x 4h bars, so divide by 6
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * (vol_ma_1d_aligned[i] / 6.0)
        
        if position == 0:
            # Look for long entry: uptrend (price > 1w EMA50) + EMA bullish crossover + volume
            if close[i] > ema_50_1w_aligned[i] and ema_9[i] > ema_21[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: downtrend (price < 1w EMA50) + EMA bearish crossover + volume
            elif close[i] < ema_50_1w_aligned[i] and ema_9[i] < ema_21[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on EMA bearish crossover or trend reversal
            if ema_9[i] < ema_21[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on EMA bullish crossover or trend reversal
            if ema_9[i] > ema_21[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals