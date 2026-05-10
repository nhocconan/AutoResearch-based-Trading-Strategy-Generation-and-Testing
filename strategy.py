#!/usr/bin/env python3
# 1h_Camarilla_R1S1_Breakout_4hTrend_1dVolume
# Hypothesis: Long when price breaks above Camarilla R1 with 4h uptrend (price > 4h EMA50) and 1d volume > 1.5x average.
# Short when price breaks below Camarilla S1 with 4h downtrend (price < 4h EMA50) and 1d volume > 1.5x average.
# Uses Camarilla pivot levels for intraday support/resistance, 4h EMA for trend filter, and 1d volume spike for confirmation.
# Designed for 15-37 trades/year on 1h timeframe to avoid fee drag.

name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dVolume"
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
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.nanmean(tr[i-13:i+1])
    
    # Get 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d OHLC for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R1, S1)
    camarilla_r1 = np.full(len(df_1d), np.nan)
    camarilla_s1 = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i]):
            continue
        camarilla_r1[i] = close_1d[i] + (high_1d[i] - low_1d[i]) * 1.0833 / 12
        camarilla_s1[i] = close_1d[i] - (high_1d[i] - low_1d[i]) * 1.0833 / 12
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 1d volume average (20 periods)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        vol_ma_1d[i] = np.nanmean(vol_1d[i-20:i])
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Long: price breaks above R1, 4h uptrend, volume spike
            if in_session and close[i] > camarilla_r1_aligned[i] and close[i-1] <= camarilla_r1_aligned[i-1] and close[i] > ema_50_4h_aligned[i] and volume[i] > 1.5 * vol_ma_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1, 4h downtrend, volume spike
            elif in_session and close[i] < camarilla_s1_aligned[i] and close[i-1] >= camarilla_s1_aligned[i-1] and close[i] < ema_50_4h_aligned[i] and volume[i] > 1.5 * vol_ma_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: price breaks below S1 or stoploss hit
            if close[i] < camarilla_s1_aligned[i] or (i > 0 and low[i] < camarilla_s1_aligned[i] - 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: price breaks above R1 or stoploss hit
            if close[i] > camarilla_r1_aligned[i] or (i > 0 and high[i] > camarilla_r1_aligned[i] + 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals