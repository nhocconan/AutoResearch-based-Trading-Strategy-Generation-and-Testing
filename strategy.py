#!/usr/bin/env python3
"""
12h_1d_1w_RSI_MeanReversion_v1
Hypothesis: On 12h timeframe, use RSI(14) on 1d for mean reversion signals (overbought/oversold),
filtered by weekly volatility regime (low volatility = trending, avoid chop).
Enter long when RSI < 30 and price > 200-period EMA on 1d (bullish bias),
enter short when RSI > 70 and price < 200-period EMA on 1d (bearish bias).
Exit when RSI returns to neutral zone (40-60).
Uses volume confirmation to avoid false signals.
Designed for low trade frequency (<30/year) by requiring multiple confluence factors.
Works in bull/bear via 1d EMA filter and RSI mean reversion.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_RSI_MeanReversion_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY RSI AND EMA200 ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # RSI(14) calculation
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    
    # Initialize first average
    if len(gain) >= 14:
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        
        for i in range(14, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.zeros_like(close_1d)
    rsi = np.zeros_like(close_1d)
    for i in range(13, len(close_1d)):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100 if avg_gain[i] > 0 else 0
    
    # EMA200 on daily
    ema200 = np.zeros_like(close_1d)
    ema200[0] = close_1d[0]
    alpha = 2 / (200 + 1)
    for i in range(1, len(close_1d)):
        ema200[i] = alpha * close_1d[i] + (1 - alpha) * ema200[i-1]
    
    # === WEEKLY VOLATILITY REGIME FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly ATR(14)
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = np.zeros_like(tr)
    for i in range(14, len(tr)):
        if i == 14:
            atr_14[i] = np.nanmean(tr[1:i+1])
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Volatility regime: low volatility = trending market
    vol_ma = np.zeros_like(atr_14)
    for i in range(len(atr_14)):
        if i >= 30:
            vol_ma[i] = np.mean(atr_14[i-29:i+1])
        else:
            vol_ma[i] = np.nan
    vol_regime = atr_14 < vol_ma  # True when low volatility (trending)
    
    # Align data to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200)
    vol_regime_aligned = align_htf_to_ltf(prices, df_1w, vol_regime.astype(float))
    
    # Volume average (20-period for 12h = ~10 days)
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        vol_avg[i] = vol_sum / vol_count if vol_count > 0 else 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if indicators not available
        if (np.isnan(rsi_aligned[i]) or np.isnan(ema200_aligned[i]) or 
            np.isnan(vol_regime_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Only trade in low volatility (trending) regime
        in_trend_regime = vol_regime_aligned[i] > 0.5
        
        # Entry conditions: RSI extremes with EMA200 bias
        long_setup = (rsi_aligned[i] < 30) and (close[i] > ema200_aligned[i]) and vol_confirm and in_trend_regime
        short_setup = (rsi_aligned[i] > 70) and (close[i] < ema200_aligned[i]) and vol_confirm and in_trend_regime
        
        # Exit conditions: RSI returns to neutral zone
        exit_long = rsi_aligned[i] > 40
        exit_short = rsi_aligned[i] < 60
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals