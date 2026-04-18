#!/usr/bin/env python3
"""
4h_Multiplier_Trend_With_RSI_Filter_v1
Hypothesis: Use a simple multiplier-based trend (price > 1.01*EMA20 for long, < 0.99*EMA20 for short) combined with RSI extremes (>70 for short, <30 for long) and volume confirmation. Works in trending markets via EMA breakout and in ranging markets via RSI mean reversion. Volume filter reduces false signals. Designed for ~25 trades/year to minimize fee drag.
"""

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
    
    # EMA20 trend filter (12h)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema20 = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20)
    
    # RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema20_12h_aligned[i]) or np.isnan(rsi[i]) or
            np.isnan(volume_spike[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_val = ema20_12h_aligned[i]
        rsi_val = rsi[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price > 1.01*EMA20, RSI < 30 (oversold), volume spike
            if price > 1.01 * ema_val and rsi_val < 30 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price < 0.99*EMA20, RSI > 70 (overbought), volume spike
            elif price < 0.99 * ema_val and rsi_val > 70 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price < 0.995*EMA20 or RSI > 70
            if price < 0.995 * ema_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price > 1.005*EMA20 or RSI < 30
            if price > 1.005 * ema_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Multiplier_Trend_With_RSI_Filter_v1"
timeframe = "4h"
leverage = 1.0