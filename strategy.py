#!/usr/bin/env python3
# 12h_7d_Momentum_RSI_4258
# Hypothesis: Uses weekly RSI(7) on daily closes to detect momentum extremes. 
# Long when weekly RSI < 40 (oversold) and price above daily EMA20; short when weekly RSI > 60 (overbought) and price below daily EMA20.
# Weekly timeframe ensures low trade frequency (<30/year) while capturing multi-week momentum reversals.
# Works in bull markets (buying dips in uptrend) and bear markets (selling rallies in downtrend) by fading extremes.
# Volume confirmation (volume > 1.5x 20-period average) reduces false signals.

name = "12h_7d_Momentum_RSI_4258"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for weekly RSI and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly RSI(7) on daily close ---
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing: alpha = 1/period
    avg_gain = pd.Series(gain).ewm(alpha=1/7, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/7, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_7 = 100 - (100 / (1 + rs))
    
    # Align weekly RSI to 12h
    rsi_7_aligned = align_htf_to_ltf(prices, df_1d, rsi_7)
    
    # --- Daily EMA20 for trend filter ---
    ema_20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    
    # --- Volume Spike Detection (20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_7_aligned[i]) or np.isnan(ema_20_aligned[i]) or
            np.isnan(vol_ratio[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: weekly RSI < 40 (oversold) and price above daily EMA20
            if (rsi_7_aligned[i] < 40 and 
                close[i] > ema_20_aligned[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: weekly RSI > 60 (overbought) and price below daily EMA20
            elif (rsi_7_aligned[i] > 60 and 
                  close[i] < ema_20_aligned[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: RSI returns to neutral zone (40-60)
            if position == 1:
                # Exit long: RSI >= 40 (no longer oversold)
                if rsi_7_aligned[i] >= 40:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: RSI <= 60 (no longer overbought)
                if rsi_7_aligned[i] <= 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals