#!/usr/bin/env python3
# 1h_Camarilla_R1S1_Breakout_4hTrend_Volume
# Hypothesis: Uses Camarilla pivot levels (R1/S1) from 4h timeframe for breakout signals.
# Goes long when price breaks above R1 with volume confirmation and 4h uptrend.
# Goes short when price breaks below S1 with volume confirmation and 4h downtrend.
# 1h timeframe used for precise entry timing, with 4h trend filter to avoid counter-trend trades.
# Volume confirmation requires volume > 1.5x 20-period average to confirm breakout strength.
# Designed to work in both bull and bear markets by aligning with higher timeframe trend.
# Targets 15-37 trades per year on 1h timeframe with position size 0.20.
# R1/S1 levels provide tighter breakouts than R2/S2, reducing false signals while capturing meaningful moves.

name = "1h_Camarilla_R1S1_Breakout_4hTrend_Volume"
timeframe = "1h"
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
    
    # Get 4h data for Camarilla pivot levels (using previous bar's data)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 4h bar's OHLC
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    # Using previous bar's data to avoid look-ahead
    prev_close = df_4h['close'].shift(1).values
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    
    # Calculate R1 and S1
    r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # Get 4h data for trend filter (EMA50)
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate volume average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Warmup for volume MA and 4h EMA
    
    for i in range(start_idx, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from 4h
        uptrend = close[i] > ema_50_4h_aligned[i]
        downtrend = close[i] < ema_50_4h_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: price breaks above R1 with volume confirmation and 4h uptrend
            if close[i] > r1_aligned[i] and volume_confirm and uptrend:
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below S1 with volume confirmation and 4h downtrend
            elif close[i] < s1_aligned[i] and volume_confirm and downtrend:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price falls below R1 or trend turns down
            if close[i] < r1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price rises above S1 or trend turns up
            if close[i] > s1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals