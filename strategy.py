#!/usr/bin/env python3
"""
4h_ChaikinMoneyFlow_Breakout_V1
Hypothesis: Use Chaikin Money Flow (CMF) on 4h to detect institutional buying/selling pressure, combined with 4h price breakout above/below Donchian(20) channels. CMF > 0.25 indicates strong buying pressure for longs; CMF < -0.25 indicates selling pressure for shorts. Designed for low trade frequency (~20-40/year) to capture strong momentum moves with institutional confirmation, working in both bull (catch breakouts) and bear (fade false breaks) markets.
"""

import numpy as np
import pandas as pd
from mtr_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Chaikin Money Flow (CMF) on 4h (20-period) ===
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # Money Flow Volume = Money Flow Multiplier * Volume
    # CMF = 20-period sum of Money Flow Volume / 20-period sum of Volume
    
    # Avoid division by zero
    high_low = high - low
    high_low[high_low == 0] = 1e-10
    
    mf_multiplier = ((close - low) - (high - close)) / high_low
    mf_volume = mf_multiplier * volume
    
    # Calculate 20-period sums
    mf_volume_sum = np.zeros_like(mf_volume)
    volume_sum = np.zeros_like(volume)
    
    for i in range(20, len(mf_volume)):
        mf_volume_sum[i] = np.sum(mf_volume[i-19:i+1])
        volume_sum[i] = np.sum(volume[i-19:i+1])
    
    # Initialize CMF array
    cmf = np.full_like(close, np.nan)
    for i in range(20, len(volume_sum)):
        if volume_sum[i] != 0:
            cmf[i] = mf_volume_sum[i] / volume_sum[i]
    
    # === Donchian Channel (20-period) for breakout ===
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    
    for i in range(20, len(high)):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # === 1-day RSI filter for trend strength ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate RSI(14) on daily
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    
    # First average
    avg_gain[13] = np.mean(gain[1:15])
    avg_loss[13] = np.mean(loss[1:15])
    
    # Subsequent averages (Wilder's smoothing)
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14[0:14] = np.nan
    
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    signals = np.zeros(n)
    
    # Warmup: enough for CMF(20), Donchian(20), and RSI(14)
    warmup = 40
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(cmf[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(rsi_14_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry conditions
        long_breakout = close[i] > highest_high[i]
        short_breakout = close[i] < lowest_low[i]
        
        # CMF thresholds for institutional pressure
        cmf_buy_pressure = cmf[i] > 0.25
        cmf_sell_pressure = cmf[i] < -0.25
        
        # RSI filter: avoid extreme overbought/oversold for better follow-through
        rsi_not_overbought = rsi_14_aligned[i] < 70
        rsi_not_oversold = rsi_14_aligned[i] > 30
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: bullish breakout + buying pressure + RSI not overbought
            if long_breakout and cmf_buy_pressure and rsi_not_overbought:
                signals[i] = 0.25
                position = 1
                continue
            # Short: bearish breakout + selling pressure + RSI not oversold
            elif short_breakout and cmf_sell_pressure and rsi_not_oversold:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit when price breaks below Donchian low OR CMF turns negative
            if close[i] < lowest_low[i] or cmf[i] < 0:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price breaks above Donchian high OR CMF turns positive
            if close[i] > highest_high[i] or cmf[i] > 0:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_ChaikinMoneyFlow_Breakout_V1"
timeframe = "4h"
leverage = 1.0