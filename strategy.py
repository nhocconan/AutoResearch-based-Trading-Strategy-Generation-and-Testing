#!/usr/bin/env python3
# 4h_1d_CCI_Reversal_TrendFilter
# Hypothesis: 4h price reverses from 1-day CCI extremes with trend filter.
# Long when: CCI(20) < -100 AND 1d EMA50 rising AND volume > 1.3x 20-bar avg.
# Short when: CCI(20) > 100 AND 1d EMA50 falling AND volume > 1.3x 20-bar avg.
# Exit when CCI crosses back above -50 (long) or below 50 (short) OR trend reverses.
# CCI captures overextended moves; EMA50 filters counter-trend in bear markets.
# Works in bull by buying dips in uptrend; works in bear by selling rallies in downtrend.
# Target: 20-30 trades/year (80-120 total over 4 years) to avoid fee drag.

name = "4h_1d_CCI_Reversal_TrendFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for CCI and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d CCI(20) ---
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    sma_tp = np.full(len(typical_price), np.nan)
    mad = np.full(len(typical_price), np.nan)
    for i in range(20, len(typical_price)):
        sma_tp[i] = np.mean(typical_price[i-20:i])
        mad[i] = np.mean(np.abs(typical_price[i-20:i] - sma_tp[i]))
    cci = np.full(len(typical_price), np.nan)
    for i in range(20, len(typical_price)):
        if mad[i] != 0:
            cci[i] = (typical_price[i] - sma_tp[i]) / (0.015 * mad[i])
        else:
            cci[i] = 0.0
    
    # --- 1d EMA50 trend ---
    close_1d = df_1d['close'].values
    ema_1d = np.full(len(close_1d), np.nan)
    for i in range(len(close_1d)):
        if i < 50:
            ema_1d[i] = np.nan
        elif i == 50:
            ema_1d[i] = np.mean(close_1d[0:50])
        else:
            ema_1d[i] = (close_1d[i] * 2 / (50 + 1)) + (ema_1d[i-1] * (49 / (50 + 1)))
    
    # EMA slope
    ema_slope_1d = np.full(len(close_1d), np.nan)
    for i in range(51, len(close_1d)):
        ema_slope_1d[i] = ema_1d[i] - ema_1d[i-1]
    
    # --- 4h volume MA(20) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 1d indicators to 4h
    cci_aligned = align_htf_to_ltf(prices, df_1d, cci)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    ema_slope_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_slope_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max(CCI needs 20, EMA50 needs 50, vol MA needs 20)
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(cci_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or
            np.isnan(ema_slope_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # CCI extremes
        cci_overbought = cci_aligned[i] > 100
        cci_oversold = cci_aligned[i] < -100
        cci_exit_long = cci_aligned[i] > -50
        cci_exit_short = cci_aligned[i] < 50
        
        # Volume confirmation
        vol_spike = volume[i] > vol_ma[i] * 1.3
        
        if position == 0:
            if cci_oversold and ema_slope_1d_aligned[i] > 0 and vol_spike:
                # Long: oversold in uptrend
                signals[i] = 0.25
                position = 1
            elif cci_overbought and ema_slope_1d_aligned[i] < 0 and vol_spike:
                # Short: overbought in downtrend
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: CCI crosses above -50 OR trend turns down
                if cci_exit_long or ema_slope_1d_aligned[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: CCI crosses below 50 OR trend turns up
                if cci_exit_short or ema_slope_1d_aligned[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals