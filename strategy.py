#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-week Exponential Moving Average (EMA50) as primary trend filter
# and 1-day Average True Range (ATR14) for volatility-based position sizing and stop management.
# Long when price > weekly EMA50 and ATR contraction (low volatility) precedes expansion.
# Short when price < weekly EMA50 and ATR contraction precedes expansion.
# Uses volume confirmation (volume > 1.3x 20-period average) to filter breakouts.
# Designed for low trade frequency (<50 trades/year) to minimize fee drag in 4h timeframe.
# Weekly EMA50 provides strong trend filter that works in both bull and bear markets.
# ATR-based sizing reduces exposure during high volatility, protecting against large drawdowns.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Get 1d data for ATR14 calculation (volatility filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day ATR14 for volatility assessment
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    # ATR contraction/expansion signal: current ATR < 0.8 * ATR 5 periods ago
    # Indicates low volatility period likely to expand (breakout setup)
    atr_ratio = atr14_1d_aligned / np.roll(atr14_1d_aligned, 5)
    atr_ratio[0:5] = 1.0  # Avoid division by zero/NaN for first 5 periods
    volatility_contraction = atr_ratio < 0.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(atr14_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price above weekly EMA50, volatility contraction, and volume confirmation
        if (close[i] > ema50_1w_aligned[i] and 
            volatility_contraction[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short entry: price below weekly EMA50, volatility contraction, and volume confirmation
        elif (close[i] < ema50_1w_aligned[i] and 
              volatility_contraction[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal or volatility expansion
        elif position == 1 and close[i] <= ema50_1w_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] >= ema50_1w_aligned[i]:
            signals[i] = 0.0
            position = 0
        # Volatility expansion exit: reduce position during high volatility
        elif position == 1 and atr_ratio[i] > 1.2:
            signals[i] = 0.125  # Reduce to half position
        elif position == -1 and atr_ratio[i] > 1.2:
            signals[i] = -0.125  # Reduce to half position
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_WeeklyEMA50_ATRVolatility_VolumeFilter"
timeframe = "4h"
leverage = 1.0