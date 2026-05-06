#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Bollinger Band squeeze + 1w Supertrend filter + volume confirmation
# Long when 1d BBWidth < 0.03 (squeeze) AND 1w Supertrend = bullish AND volume > 1.5 * avg_volume(20) on 12h
# Short when 1d BBWidth < 0.03 (squeeze) AND 1w Supertrend = bearish AND volume > 1.5 * avg_volume(20) on 12h
# Exit when price crosses 1d EMA20 (mean reversion to middle band)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Bollinger Band squeeze identifies low volatility periods primed for breakout
# 1w Supertrend filter ensures we trade with the dominant weekly trend
# Volume confirmation (1.5x) validates breakout strength while limiting overtrading
# Works in both bull (buy breakouts) and bear (sell breakdowns) markets

name = "12h_1dBBSqueeze_1wSupertrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Bollinger Band calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need at least 20 completed 1d bars for BBands
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Bollinger Bands (20, 2)
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + (2 * std_20_1d)
    lower_bb_1d = sma_20_1d - (2 * std_20_1d)
    bb_width_1d = (upper_bb_1d - lower_bb_1d) / sma_20_1d
    # Handle division by zero
    bb_width_1d = np.where(sma_20_1d == 0, 1.0, bb_width_1d)
    
    # Align 1d BBWidth to 12h timeframe (wait for completed 1d bar)
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width_1d)
    
    # Get 1w data ONCE before loop for Supertrend calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:  # Need at least 10 completed weekly bars for Supertrend
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Supertrend (10, 3.0)
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = pd.Series(high_1w - low_1w)
    tr2 = pd.Series(np.abs(high_1w - pd.Series(close_1w).shift(1)))
    tr3 = pd.Series(np.abs(low_1w - pd.Series(close_1w).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Basic Upper and Lower Bands
    hl_avg = (high_1w + low_1w) / 2
    upper_basic = hl_avg + (multiplier * atr)
    lower_basic = hl_avg - (multiplier * atr)
    
    # Final Upper and Lower Bands
    upper_band = np.zeros_like(close_1w)
    lower_band = np.zeros_like(close_1w)
    upper_band[0] = upper_basic[0]
    lower_band[0] = lower_basic[0]
    
    for i in range(1, len(close_1w)):
        if upper_basic[i] < upper_band[i-1] or close_1w[i-1] > upper_band[i-1]:
            upper_band[i] = upper_basic[i]
        else:
            upper_band[i] = upper_band[i-1]
            
        if lower_basic[i] > lower_band[i-1] or close_1w[i-1] < lower_band[i-1]:
            lower_band[i] = lower_basic[i]
        else:
            lower_band[i] = lower_band[i-1]
    
    # Supertrend
    supertrend = np.zeros_like(close_1w)
    supertrend[0] = upper_band[0]
    direction = np.ones_like(close_1w, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_1w)):
        if close_1w[i] > upper_band[i-1]:
            direction[i] = 1
        elif close_1w[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Supertrend signal: 1 = bullish, -1 = bearish
    supertrend_signal = direction
    supertrend_signal_aligned = align_htf_to_ltf(prices, df_1w, supertrend_signal)
    
    # Calculate 1d EMA20 for exit signal
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 12h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(bb_width_aligned[i]) or np.isnan(supertrend_signal_aligned[i]) or 
            np.isnan(ema_20_1d_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: BB squeeze, 1w Supertrend bullish, volume spike, in session
            if (bb_width_aligned[i] < 0.03 and 
                supertrend_signal_aligned[i] == 1 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: BB squeeze, 1w Supertrend bearish, volume spike, in session
            elif (bb_width_aligned[i] < 0.03 and 
                  supertrend_signal_aligned[i] == -1 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses above 1d EMA20 (mean reversion)
            if close[i] > ema_20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses below 1d EMA20 (mean reversion)
            if close[i] < ema_20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals