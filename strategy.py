#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend direction with 1w volatility filter and volume confirmation.
# Uses Kaufman's Adaptive Moving Average (KAMA) for trend identification,
# weekly Bollinger Band width for volatility regime, and volume spikes for entry confirmation.
# Designed to work in both bull (expanding volatility + uptrend) and bear (expanding volatility + downtrend) markets.
# Target: 10-20 trades/year to minimize fee drag while capturing major moves.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for KAMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate KAMA (10, 2, 30) on daily close
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # will fix below
    
    # Proper volatility calculation: sum of absolute changes over 10 periods
    volatility = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        volatility[i] = np.sum(np.abs(np.diff(close_1d[i-10:i+1])))
    
    # Avoid division by zero
    er = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if volatility[i] > 0:
            er[i] = np.abs(close_1d[i] - close_1d[i-10]) / volatility[i] if i >= 10 else 0
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if np.isnan(kama[i-1]):
            kama[i] = close_1d[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Get weekly data for Bollinger Band width (volatility filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate Bollinger Band width (20, 2) on weekly close
    bb_width = np.zeros_like(close_1w)
    for i in range(20, len(close_1w)):
        ma = np.mean(close_1w[i-20:i+1])
        std = np.std(close_1w[i-20:i+1])
        if ma > 0:
            bb_width[i] = (2 * std) / ma  # normalized width
        else:
            bb_width[i] = 0
    
    # Bollinger Band width needs 1 extra weekly bar for confirmation
    bb_width_aligned = align_htf_to_ltf(prices, df_1w, bb_width, additional_delay_bars=1)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for indicators
    start_idx = max(20, 30)
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(bb_width_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from daily KAMA
        # Use previous bar's KAMA to avoid look-ahead
        if i > 0 and not np.isnan(kama_aligned[i-1]):
            trend_up = close[i] > kama_aligned[i]  # price above KAMA = uptrend
            trend_down = close[i] < kama_aligned[i]  # price below KAMA = downtrend
        else:
            trend_up = False
            trend_down = False
        
        # Volatility filter: expanding volatility (BB width increasing)
        if i > 0 and not np.isnan(bb_width_aligned[i-1]):
            vol_expanding = bb_width_aligned[i] > bb_width_aligned[i-1]
        else:
            vol_expanding = False
        
        if position == 0:
            # Long entry: price above KAMA + expanding volatility + volume spike
            if (trend_up and 
                vol_expanding and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price below KAMA + expanding volatility + volume spike
            elif (trend_down and 
                  vol_expanding and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below KAMA or volatility contracts
            if (not trend_up or 
                not vol_expanding):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA or volatility contracts
            if (not trend_down or 
                not vol_expanding):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_1wBBWidth_Volume_v1"
timeframe = "1d"
leverage = 1.0