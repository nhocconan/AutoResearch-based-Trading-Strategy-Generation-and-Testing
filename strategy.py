#!/usr/bin/env python3
# 4h_Camarilla_R1S1_Breakout_1dTrend_Volume_Filtered_v4
# Hypothesis: Uses Camarilla R1/S1 breakouts with 1d EMA34 trend filter and volume confirmation.
# Adds a momentum filter (price > VWAP) to improve win rate and reduce false breakouts.
# Targets 15-25 trades/year to minimize fee drag. Designed to work in both bull and bear markets by
# aligning with the 1d trend and requiring volume confirmation to avoid low-probability entries.
# Position size 0.25 for balanced risk.

name = "4h_Camarilla_R1S1_Breakout_1dTrend_Volume_Filtered_v4"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data for Camarilla pivot levels (using previous day's data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate VWAP for momentum filter
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    
    # Calculate Camarilla levels from previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate R1 and S1 (tighter levels than R2/S2)
    r1 = prev_close + (prev_high - prev_low) * 1.1 / 6
    s1 = prev_close - (prev_high - prev_low) * 1.1 / 6
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Get 1d data for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34, 14)  # Warmup for volume MA, 1d EMA, and ATR
    
    for i in range(start_idx, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i]) or np.isnan(vwap[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from 1d
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Stronger volume confirmation and volatility filter
        volume_confirm = volume[i] > volume_ma[i] * 2.5
        volatility_filter = atr[i] > 0  # Ensure valid ATR
        
        # Momentum filter: price above/below VWAP
        above_vwap = close[i] > vwap[i]
        below_vwap = close[i] < vwap[i]
        
        if position == 0:
            # Long entry: price breaks above R1 with volume confirmation, 1d uptrend, volatility, and above VWAP
            if close[i] > r1_aligned[i] and volume_confirm and uptrend and volatility_filter and above_vwap:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 with volume confirmation, 1d downtrend, volatility, and below VWAP
            elif close[i] < s1_aligned[i] and volume_confirm and downtrend and volatility_filter and below_vwap:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below R1 or trend turns down
            if close[i] < r1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above S1 or trend turns up
            if close[i] > s1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals