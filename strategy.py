#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dATR_Trend_VolumeSpike
Hypothesis: 4-hour Camarilla R1/S1 breakout with 1-day ATR trend filter and volume confirmation.
Targets 20-30 trades/year by requiring: 1) price breaks daily R1/S1 levels, 2) aligned with 1d ATR-based trend (above/below ATR-weighted average),
3) volume > 1.8x 20-period average. Uses discrete position sizing (0.25) to minimize fee churn.
ATR trend filter adapts to volatility, working in both bull and bear markets by following the dominant trend regime.
Volume spike filter reduces false breakouts. Designed for BTC/ETH with SOL as secondary.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for Camarilla pivots and ATR trend (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla R1 and S1 levels (R1 = C + 1.1*(HL/4), S1 = C - 1.1*(HL/4))
    R1 = prev_close + 1.1 * prev_range * (1.0/4.0)
    S1 = prev_close - 1.1 * prev_range * (1.0/4.0)
    
    # 1d ATR(14) for trend filter
    tr1 = np.maximum(df_1d['high'].values, np.roll(df_1d['close'].values, 1)) - np.minimum(df_1d['low'].values, np.roll(df_1d['close'].values, 1))
    tr2 = np.abs(np.roll(df_1d['close'].values, 1) - df_1d['close'].values)
    tr = np.maximum(tr1, tr2)
    tr[0] = tr1[0]  # first period
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # ATR trend: price > close + 0.5*ATR = uptrend, price < close - 0.5*ATR = downtrend
    atr_trend_up = df_1d['close'].values + 0.5 * atr_14
    atr_trend_down = df_1d['close'].values - 0.5 * atr_14
    
    # Align 1d levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    atr_trend_up_aligned = align_htf_to_ltf(prices, df_1d, atr_trend_up)
    atr_trend_down_aligned = align_htf_to_ltf(prices, df_1d, atr_trend_down)
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d previous data (1) + ATR14 (14) + volume MA (20)
    start_idx = 20 + 14 + 1  # Conservative warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(atr_trend_up_aligned[i]) or np.isnan(atr_trend_down_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to 1d ATR bands
        uptrend = curr_close > atr_trend_up_aligned[i]
        downtrend = curr_close < atr_trend_down_aligned[i]
        
        if position == 0:
            # Look for entry signals with volume confirmation and trend alignment
            # Long breakout: price breaks above R1 with uptrend and volume confirmation
            long_breakout = (curr_close > R1_aligned[i]) and uptrend and volume_confirm[i]
            # Short breakout: price breaks below S1 with downtrend and volume confirmation
            short_breakout = (curr_close < S1_aligned[i]) and downtrend and volume_confirm[i]
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit conditions
            # Exit if price breaks below S1 (mean reversion) or trend changes to downtrend
            if curr_close < S1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Exit if price breaks above R1 (mean reversion) or trend changes to uptrend
            if curr_close > R1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dATR_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0