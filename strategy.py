#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: 12h chart strategy using Camarilla pivot levels (R3/S3) breakouts filtered by daily EMA34 trend and volume confirmation.
# Long when price breaks above Camarilla R3 with volume > 1.5x average and price > daily EMA34.
# Short when price breaks below Camarilla S3 with volume > 1.5x average and price < daily EMA34.
# Exit when price returns to the Camarilla pivot level (CP).
# Uses weekly trend filter for additional bias in both bull and bear markets.
# Target: 15-35 trades/year per symbol to minimize fee drift.

timeframe = "12h"
name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
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
    
    # Get daily data for Camarilla levels and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # R3 = Close + 1.1*(High - Low)/2
    # S3 = Close - 1.1*(High - Low)/2
    # CP = (High + Low + Close)/3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 2.0
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 2.0
    camarilla_cp = (high_1d + low_1d + close_1d) / 3.0
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_cp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_cp)
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get weekly data for trend bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Weekly EMA34 for bias
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: 1.5x average volume (2-period = 1 day on 12h chart)
    vol_ma = pd.Series(volume).rolling(window=2, min_periods=2).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 2)  # Ensure we have EMA and volume data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_cp_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with volume, price > daily EMA34, and weekly bias bullish
            if (high[i] > camarilla_r3_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i] and 
                close[i] > ema_34_aligned[i] and
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume, price < daily EMA34, and weekly bias bearish
            elif (low[i] < camarilla_s3_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i] and 
                  close[i] < ema_34_aligned[i] and
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to pivot level (mean reversion)
            if low[i] <= camarilla_cp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to pivot level (mean reversion)
            if high[i] >= camarilla_cp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals