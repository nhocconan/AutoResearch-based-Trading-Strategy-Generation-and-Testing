#!/usr/bin/env python3
"""
12h_RSI20_Overbought_Oversold_1dTrend_Volume
Hypothesis: On 12h timeframe, use RSI(20) with oversold (<30) and overbought (>70) levels for entries,
filtered by 1d trend (price above/below 50 EMA) and volume confirmation (volume > 1.5x 20-period average).
Designed for low trade frequency (~15-25 trades/year) to minimize fee drag while capturing reversals
in both bull and bear markets. Uses strict entry conditions to avoid overtrading.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 12h RSI(20) ===
    delta = np.diff(prices['close'].values)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full_like(prices['close'].values, np.nan)
    avg_loss = np.full_like(prices['close'].values, np.nan)
    # Wilder's smoothing
    avg_gain[19] = np.nanmean(gain[1:20])
    avg_loss[19] = np.nanmean(loss[1:20])
    for i in range(20, len(delta)+1):
        avg_gain[i] = (avg_gain[i-1] * 19 + gain[i-1]) / 20
        avg_loss[i] = (avg_loss[i-1] * 19 + loss[i-1]) / 20
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Volume confirmation: 12h volume > 1.5x 20-period average ===
    vol = prices['volume'].values
    vol_ma = pd.Series(vol).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, vol / vol_ma, 0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(rsi[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        rsi_val = rsi[i]
        ema_50_val = ema_50_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Long: RSI < 30 (oversold), price above 1d EMA50 (uptrend), volume confirmation
            if (rsi_val < 30 and 
                price_close > ema_50_val and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought), price below 1d EMA50 (downtrend), volume confirmation
            elif (rsi_val > 70 and 
                  price_close < ema_50_val and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when RSI returns to neutral zone (40-60)
            if position == 1 and rsi_val >= 40:
                signals[i] = 0.0
                position = 0
            elif position == -1 and rsi_val <= 60:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_RSI20_Overbought_Oversold_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0