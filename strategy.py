#!/usr/bin/env python3
# 1h_4h1D_Trend_Momentum_With_Volume_v2
# Hypothesis: Use 4h for trend direction (EMA21) and 1d for momentum (RSI50 > 50), with volume confirmation on 1h.
# Enter long when 4h EMA21 rising, 1d RSI > 50, and 1h price breaks above 20-period high with volume > 1.5x average.
# Enter short when 4h EMA21 falling, 1d RSI < 50, and 1h price breaks below 20-period low with volume > 1.5x average.
# Exit when trend reverses or momentum fails. Designed for low trade frequency (15-30/year) to avoid fee drag.
# Works in bull (follows 4h trend) and bear (avoids counter-trend trades via 4h EMA filter).

name = "1h_4h1D_Trend_Momentum_With_Volume_v2"
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
    
    # === 4h Trend: EMA21 ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 4h EMA slope (rising/falling)
    ema_4h_slope = np.zeros_like(ema_4h_aligned)
    ema_4h_slope[1:] = ema_4h_aligned[1:] - ema_4h_aligned[:-1]
    
    # === 1d Momentum: RSI(14) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === 1h Breakout: 20-period high/low ===
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1h Volume: 20-period average ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 21, 14)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any key data is NaN
        if np.isnan(ema_4h_aligned[i]) or np.isnan(ema_4h_slope[i]) or \
           np.isnan(rsi_1d_aligned[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: 4h EMA rising, 1d RSI > 50, break above 20-period high with volume
            if ema_4h_slope[i] > 0 and rsi_1d_aligned[i] > 50 and close[i] > high_20[i] and vol_confirm:
                signals[i] = 0.20
                position = 1
            # Short: 4h EMA falling, 1d RSI < 50, break below 20-period low with volume
            elif ema_4h_slope[i] < 0 and rsi_1d_aligned[i] < 50 and close[i] < low_20[i] and vol_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: 4h EMA falling OR 1d RSI < 50
            if ema_4h_slope[i] < 0 or rsi_1d_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: 4h EMA rising OR 1d RSI > 50
            if ema_4h_slope[i] > 0 or rsi_1d_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals