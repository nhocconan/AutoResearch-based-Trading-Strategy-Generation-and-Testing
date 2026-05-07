#!/usr/bin/env python3

name = "4h_Combo_Reversal_With_Volume_And_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for RSI and trend ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 1d RSI (14)
    close_1d = pd.Series(df_1d['close'].values)
    delta = close_1d.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_14_1d = (100 - (100 / (1 + rs))).values
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # 1d EMA50 for trend
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 4h RSI (14) for mean reversion ===
    close_4h = pd.Series(close)
    delta_4h = close_4h.diff()
    gain_4h = delta_4h.clip(lower=0)
    loss_4h = -delta_4h.clip(upper=0)
    avg_gain_4h = gain_4h.rolling(window=14, min_periods=14).mean()
    avg_loss_4h = loss_4h.rolling(window=14, min_periods=14).mean()
    rs_4h = avg_gain_4h / avg_loss_4h.replace(0, np.nan)
    rsi_14_4h = (100 - (100 / (1 + rs_4h))).values
    
    # === Volume filter: current > 2.0x 20-period average ===
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 6  # ~1 day for 4h
    
    start_idx = max(20, 14, 50)  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_14_1d_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(rsi_14_4h[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: 1d RSI oversold (<30) + 4h RSI oversold (<30) + uptrend (price > EMA50) + volume
            if (rsi_14_1d_aligned[i] < 30 and 
                rsi_14_4h[i] < 30 and 
                close[i] > ema_50_1d_aligned[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: 1d RSI overbought (>70) + 4h RSI overbought (>70) + downtrend (price < EMA50) + volume
            elif (rsi_14_1d_aligned[i] > 70 and 
                  rsi_14_4h[i] > 70 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: 4h RSI overbought (>70) OR trend change
            if rsi_14_4h[i] > 70 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: 4h RSI oversold (<30) OR trend change
            if rsi_14_4h[i] < 30 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Mean reversion at RSI extremes (14) on both 1d and 4h timeframes,
# filtered by higher timeframe trend (EMA50) and volume confirmation.
# Works in bull markets (buying dips in uptrend) and bear markets (selling rallies in downtrend).
# Low trade frequency due to strict double RSI + trend + volume requirements.