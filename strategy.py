#!/usr/bin/env python3
# 1h_4H1D_TrendWithVol_CamarillaBreakout
# Hypothesis: Use 4h trend (EMA50) and 1d momentum (close vs open) as directional filter, 
# then trade 1h breakouts of prior 4h Camarilla R3/S3 levels with volume confirmation.
# Trend filter reduces whipsaw in sideways markets; volume avoids low-conviction breakouts.
# Works in bull (breakouts follow trend) and bear (mean reversion at extremes via trend filter).
# Target: 15-30 trades/year (~60-120 over 4 years) to stay under fee drag.

name = "1h_4H1D_TrendWithVol_CamarillaBreakout"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 4h Trend Filter: EMA50 ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 1d Momentum Filter: close > open (bullish day) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    open_1d = df_1d['open'].values
    close_1d = df_1d['close'].values
    bullish_day = close_1d > open_1d  # True if bullish daily candle
    bearish_day = close_1d < open_1d  # True if bearish daily candle
    
    bullish_day_aligned = align_htf_to_ltf(prices, df_1d, bullish_day.astype(float))
    bearish_day_aligned = align_htf_to_ltf(prices, df_1d, bearish_day.astype(float))
    
    # === 4h Camarilla Levels (R3, S3) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Prior 4h bar's OHLC for Camarilla calculation
    prev_high_4h = high_4h.shift(1).values
    prev_low_4h = low_4h.shift(1).values
    prev_close_4h = close_4h.shift(1).values
    
    # Calculate Camarilla levels
    R3 = prev_close_4h + (prev_high_4h - prev_low_4h) * 1.1 / 4
    S3 = prev_close_4h - (prev_high_4h - prev_low_4h) * 1.1 / 4
    
    # Align to 1h timeframe (wait for 4h bar close)
    R3_aligned = align_htf_to_ltf(prices, df_4h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_4h, S3)
    
    # === Volume Confirmation (24-period average) ===
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(vol_ma_24[i]) or
            np.isnan(bullish_day_aligned[i]) or np.isnan(bearish_day_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend direction from 4h EMA50
        trend_up = close[i] > ema_50_4h_aligned[i]
        trend_down = close[i] < ema_50_4h_aligned[i]
        
        # Momentum from 1d candle
        mom_bull = bullish_day_aligned[i] > 0.5
        mom_bear = bearish_day_aligned[i] > 0.5
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_24[i]
        
        if position == 0:
            # LONG: Price breaks above R3 with volume, bullish 4h trend, bullish day
            if (close[i] > R3_aligned[i] and vol_ok and trend_up and mom_bull):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S3 with volume, bearish 4h trend, bearish day
            elif (close[i] < S3_aligned[i] and vol_ok and trend_down and mom_bear):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # EXIT LONG: Price returns to S3 or trend/momentum changes
            if (close[i] < S3_aligned[i] or not trend_up or not mom_bull):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price returns to R3 or trend/momentum changes
            if (close[i] > R3_aligned[i] or not trend_down or not mom_bear):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals