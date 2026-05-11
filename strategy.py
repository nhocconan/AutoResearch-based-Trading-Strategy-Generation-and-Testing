#!/usr/bin/env python3
# 1h_RSI_MeanReversion_4hTrend
# Hypothesis: Mean reversion on 1h RSI extremes, filtered by 4h EMA trend and volume spike.
# Long when: 1h RSI < 30 (oversold), 4h EMA20 rising, volume > 1.5x 20-period avg.
# Short when: 1h RSI > 70 (overbought), 4h EMA20 falling, volume > 1.5x 20-period avg.
# Exit when RSI returns to neutral (40-60) or 4h EMA trend reverses.
# Works in bull markets by buying dips in uptrend and in bear by selling rallies in downtrend.
# RSI provides mean reversion signals, EMA20 filters counter-trend moves, volume confirms strength.

name = "1h_RSI_MeanReversion_4hTrend"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Get 4h data for EMA20 trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 1h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1h RSI(14) ---
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[0:15])
            avg_loss[i] = np.mean(loss[0:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.full(n, np.nan)
    rsi = np.full(n, np.nan)
    for i in range(14, n):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100  # Avoid division by zero
    
    # --- 4h EMA20 trend ---
    close_4h = df_4h['close'].values
    ema_4h = np.full(len(close_4h), np.nan)
    for i in range(20, len(close_4h)):
        if i == 20:
            ema_4h[i] = np.mean(close_4h[0:20])
        else:
            ema_4h[i] = (close_4h[i] * 2 / (20 + 1)) + (ema_4h[i-1] * (19 / (20 + 1)))
    
    # EMA slope (rising/falling)
    ema_slope = np.full(len(close_4h), np.nan)
    for i in range(21, len(close_4h)):
        ema_slope[i] = ema_4h[i] - ema_4h[i-1]
    
    # Align 4h EMA and slope to 1h
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    ema_slope_aligned = align_htf_to_ltf(prices, df_4h, ema_slope)
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for RSI(14), EMA20, and volume MA(20)
    start_idx = max(14, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(rsi[i]) or
            np.isnan(ema_4h_aligned[i]) or
            np.isnan(ema_slope_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # RSI conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_neutral = (rsi[i] >= 40) & (rsi[i] <= 60)
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.5  # 50% above average
        
        if position == 0:
            if rsi_oversold and ema_slope_aligned[i] > 0 and vol_spike:
                # Long: oversold + rising EMA20 + volume spike
                signals[i] = 0.20
                position = 1
            elif rsi_overbought and ema_slope_aligned[i] < 0 and vol_spike:
                # Short: overbought + falling EMA20 + volume spike
                signals[i] = -0.20
                position = -1
        else:
            if position == 1:
                # Exit long: RSI returns to neutral OR EMA slope turns negative
                if rsi_neutral[i] or ema_slope_aligned[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: RSI returns to neutral OR EMA slope turns positive
                if rsi_neutral[i] or ema_slope_aligned[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals