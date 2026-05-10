#!/usr/bin/env python3
# 1h_Camarilla_R1S1_Breakout_4hTrend_Volume_Precision
# Hypothesis: Uses 4h trend (EMA34) for directional bias, 1h for timing entries at daily Camarilla R1/S1 breakouts with volume confirmation.
# Targets 15-30 trades/year by combining 4h trend filter with tight 1h entry conditions. Works in bull/bear by following 4h trend.
# Session filter (08-20 UTC) reduces noise trades. Position size 0.20 for controlled risk.

name = "1h_Camarilla_R1S1_Breakout_4hTrend_Volume_Precision"
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
    
    # Get daily data for Camarilla pivot levels (using previous day's data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate ATR for volatility filter (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels from previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate R1 and S1 (tighter levels)
    r1 = prev_close + (prev_high - prev_low) * 1.1 / 6
    s1 = prev_close - (prev_high - prev_low) * 1.1 / 6
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Get 4h EMA34 for trend filter
    ema_34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate volume average for confirmation (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34, 14)  # Warmup for volume MA, 4h EMA, and ATR
    
    for i in range(start_idx, n):
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema_34_4h_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from 4h
        uptrend = close[i] > ema_34_4h_aligned[i]
        downtrend = close[i] < ema_34_4h_aligned[i]
        
        # Volume confirmation and volatility filter
        volume_confirm = volume[i] > volume_ma[i] * 2.0
        volatility_filter = atr[i] > 0
        
        if position == 0:
            # Long entry: price breaks above R1 with volume confirmation, 4h uptrend
            if close[i] > r1_aligned[i] and volume_confirm and uptrend and volatility_filter:
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below S1 with volume confirmation, 4h downtrend
            elif close[i] < s1_aligned[i] and volume_confirm and downtrend and volatility_filter:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price falls below R1 or 4h trend turns down
            if close[i] < r1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price rises above S1 or 4h trend turns up
            if close[i] > s1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals