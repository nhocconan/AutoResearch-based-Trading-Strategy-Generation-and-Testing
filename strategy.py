#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_ADX_RSI_Momentum_with_Volume_Confirmation_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly and daily data once
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # === Weekly RSI for trend filter ===
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close_1w)
    avg_loss = np.zeros_like(close_1w)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    for i in range(15, len(close_1w)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w[0:14] = np.nan
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # === Daily ADX for trend strength ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_1d[0] - low_1d[0]
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    atr = np.zeros_like(close_1d)
    atr[0] = tr[0]
    for i in range(1, len(close_1d)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 12h RSI for momentum ===
    delta_12h = np.diff(close)
    gain_12h = np.where(delta_12h > 0, delta_12h, 0)
    loss_12h = np.where(delta_12h < 0, -delta_12h, 0)
    avg_gain_12h = np.zeros_like(close)
    avg_loss_12h = np.zeros_like(close)
    avg_gain_12h[14] = np.mean(gain_12h[1:15])
    avg_loss_12h[14] = np.mean(loss_12h[1:15])
    for i in range(15, len(close)):
        avg_gain_12h[i] = (avg_gain_12h[i-1] * 13 + gain_12h[i]) / 14
        avg_loss_12h[i] = (avg_loss_12h[i-1] * 13 + loss_12h[i]) / 14
    rs_12h = np.where(avg_loss_12h != 0, avg_gain_12h / avg_loss_12h, 0)
    rsi_12h = 100 - (100 / (1 + rs_12h))
    rsi_12h[0:14] = np.nan
    
    # === 12h Volume filter: current volume > 20-period average ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 14  # warmup for RSI
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(rsi_12h[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Trend filter: weekly RSI > 50 for bullish, < 50 for bearish
            # Momentum: 12h RSI > 55 for long, < 45 for short
            # Volume confirmation
            if (rsi_1w_aligned[i] > 50 and  # Weekly bullish trend
                rsi_12h[i] > 55 and         # 12h momentum bullish
                adx_aligned[i] > 25 and     # Strong trend
                volume[i] > vol_ma20[i]):   # Volume confirmation
                signals[i] = 0.25
                position = 1
            elif (rsi_1w_aligned[i] < 50 and  # Weekly bearish trend
                  rsi_12h[i] < 45 and         # 12h momentum bearish
                  adx_aligned[i] > 25 and     # Strong trend
                  volume[i] > vol_ma20[i]):   # Volume confirmation
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: momentum weakness or trend reversal
            if (rsi_12h[i] < 50 or      # Momentum faded
                rsi_1w_aligned[i] < 45): # Weekly trend weakening
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: momentum weakness or trend reversal
            if (rsi_12h[i] > 50 or      # Momentum faded
                rsi_1w_aligned[i] > 55): # Weekly trend weakening
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Multi-timeframe momentum strategy using weekly RSI for trend direction,
# daily ADX for trend strength, and 12h RSI for entry timing with volume confirmation.
# Enters long when weekly trend is bullish (RSI>50), 12h momentum is strong (RSI>55),
# trend is strong (ADX>25), and volume is above average. Reverses for shorts.
# Designed to work in both bull and bear markets by following the weekly trend.
# Uses discrete sizing (0.25) to minimize fee churn. Targets 50-150 trades over 4 years.