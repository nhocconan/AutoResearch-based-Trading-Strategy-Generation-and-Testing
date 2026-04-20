#!/usr/bin/env python3
# 4h_Keltner_Channel_RSI_Strategy
# Hypothesis: Keltner Channel breakouts combined with RSI momentum and volume confirmation
# capture sustained trends in both bull and bear markets. Keltner Channels adapt to volatility,
# RSI filters for momentum strength, and volume ensures institutional participation.
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.

name = "4h_Keltner_Channel_RSI_Strategy"
timeframe = "4h"
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
    
    # Get 1d data for ATR calculation (more stable volatility measure)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate ATR (10-period) on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR (10-period)
    atr_1d = pd.Series(tr_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate EMA (20-period) for Keltner Channel middle line
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel bands (20 EMA ± 2 * ATR)
    kc_upper = ema_20 + 2 * atr_1d_aligned
    kc_lower = ema_20 - 2 * atr_1d_aligned
    
    # RSI (14-period) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Ensure EMA and RSI are calculated
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or
            np.isnan(rsi[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price closes above Keltner upper + RSI > 55 + volume confirmation
            if close[i] > kc_upper[i] and rsi[i] > 55 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price closes below Keltner lower + RSI < 45 + volume confirmation
            elif close[i] < kc_lower[i] and rsi[i] < 45 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price closes below Keltner middle or RSI weakens
            if close[i] < ema_20[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price closes above Keltner middle or RSI weakens
            if close[i] > ema_20[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals