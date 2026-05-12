# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# 1h_RSI_MeanReversion_4hTrend_1dVolume
# Hypothesis: Mean reversion in 1h RSI with 4h trend filter and daily volume confirmation.
# Long when RSI < 30 in 1h, price > 4h EMA50, and daily volume > 1.5x 20-day average.
# Short when RSI > 70 in 1h, price < 4h EMA50, and daily volume > 1.5x 20-day average.
# Exit when RSI crosses back to neutral (40 for long, 60 for short).
# Designed to capture reversals in ranging markets while avoiding counter-trend trades.
# Targets 15-35 trades/year to minimize fee drag.

name = "1h_RSI_MeanReversion_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ema = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    loss_ema = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = gain_ema / (loss_ema + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Daily volume confirmation: 20-period moving average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(rsi[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: RSI oversold, price above 4h EMA50, high volume
            if rsi[i] < 30 and close[i] > ema50_4h_aligned[i] and volume[i] > vol_ma_1d_aligned[i] * 1.5:
                signals[i] = 0.20
                position = 1
            # SHORT: RSI overbought, price below 4h EMA50, high volume
            elif rsi[i] > 70 and close[i] < ema50_4h_aligned[i] and volume[i] > vol_ma_1d_aligned[i] * 1.5:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI returns to neutral territory
            if rsi[i] > 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: RSI returns to neutral territory
            if rsi[i] < 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals