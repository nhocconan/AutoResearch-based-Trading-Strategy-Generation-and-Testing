#!/usr/bin/env python3
# 1h_Choppiness_Trend_Filter
# Hypothesis: Use 4h trend direction (EMA21) and 1d choppiness regime to filter entries. Only trade in trending markets (CHOP < 38.2) in direction of 4h EMA21. Enter on 1h pullbacks to EMA21 with volume confirmation. Avoids whipsaws in ranging markets. Works in bull/bear by adapting to regime. Targets 20-40 trades/year.

name = "1h_Choppiness_Trend_Filter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # Get 1d data for choppiness regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 4h EMA21 for trend direction
    ema_21_4h = pd.Series(df_4h['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # 1d Choppiness Index (CHOP)
    atr_1d = []
    tr1 = np.maximum(df_1d['high'].values[1:], df_1d['close'].values[:-1]) - np.minimum(df_1d['low'].values[1:], df_1d['close'].values[:-1])
    tr2 = np.abs(df_1d['high'].values[1:] - df_1d['close'].values[:-1])
    tr3 = np.abs(df_1d['low'].values[1:] - df_1d['close'].values[:-1])
    tr = np.concatenate([[np.max([df_1d['high'].values[0] - df_1d['low'].values[0], 0])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Prevent division by zero
    atr_1d_safe = np.where(atr_1d == 0, 1e-10, atr_1d)
    sum_high_low_14 = pd.Series(df_1d['high'].values - df_1d['low'].values).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_high_low_14 / (atr_1d_safe * 14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 1h EMA21 for entry timing
    ema_21_1h = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume spike filter (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(ema_21_1h[i]) or np.isnan(volume_spike[i]) or np.isnan(in_session[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade in trending regime (CHOP < 38.2) and during session
        if chop_aligned[i] < 38.2 and in_session[i]:
            if position == 0:
                # Long: price pulls back to EMA21 in uptrend with volume
                if close[i] > ema_21_1h[i] and close[i] > ema_21_4h_aligned[i] and volume_spike[i]:
                    signals[i] = 0.20
                    position = 1
                # Short: price pulls back to EMA21 in downtrend with volume
                elif close[i] < ema_21_1h[i] and close[i] < ema_21_4h_aligned[i] and volume_spike[i]:
                    signals[i] = -0.20
                    position = -1
            elif position == 1:
                # Exit: trend change or chop increases
                if close[i] < ema_21_1h[i] or chop_aligned[i] >= 38.2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit: trend change or chop increases
                if close[i] > ema_21_1h[i] or chop_aligned[i] >= 38.2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
        else:
            # Range regime or outside session: flatten
            if position != 0:
                signals[i] = 0.0
                position = 0
    
    return signals