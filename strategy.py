#!/usr/bin/env python3
# 4h_Camarilla_R1S1_Breakout_1dTrend_Volume
# Hypothesis: Long when price breaks above Camarilla R1 with volume > 1.5x average in uptrend (price > 1d EMA34).
# Short when price breaks below Camarilla S1 with volume > 1.5x average in downtrend (price < 1d EMA34).
# Exit when price crosses opposite Camarilla level (S1 for long, R1 for short) or ATR-based stoploss hit.
# Uses Camarilla pivot levels for institutional support/resistance, works in both bull and bear markets by following 1d trend.
# Designed for 20-50 trades/year to avoid fee drag.

name = "4h_Camarilla_R1S1_Breakout_1dTrend_Volume"
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
    
    # Calculate ATR(20) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(20, n):
        atr[i] = np.nanmean(tr[i-19:i+1])
    
    # Get 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We need previous day's high, low, close
    df_1d_open = df_1d['open'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    camarilla_R1 = np.full(len(df_1d), np.nan)
    camarilla_S1 = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        high_low = df_1d_high[i-1] - df_1d_low[i-1]
        camarilla_R1[i] = df_1d_close[i-1] + high_low * 1.1 / 12
        camarilla_S1[i] = df_1d_close[i-1] - high_low * 1.1 / 12
    
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.nanmean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Trade only in direction of 1d EMA34 trend
            if close[i] > ema_34_1d_aligned[i]:  # Uptrend
                # Long: Price breaks above Camarilla R1 with volume confirmation
                if close[i] > camarilla_R1_aligned[i] and close[i-1] <= camarilla_R1_aligned[i-1] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = 0.25
                    position = 1
            else:  # Downtrend
                # Short: Price breaks below Camarilla S1 with volume confirmation
                if close[i] < camarilla_S1_aligned[i] and close[i-1] >= camarilla_S1_aligned[i-1] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: Price crosses below Camarilla S1 or stoploss hit
            if close[i] < camarilla_S1_aligned[i] or (i > 0 and low[i] < camarilla_S1_aligned[i] - 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price crosses above Camarilla R1 or stoploss hit
            if close[i] > camarilla_R1_aligned[i] or (i > 0 and high[i] > camarilla_R1_aligned[i] + 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals