#!/usr/bin/env python3
name = "4h_RSI2_MeanReversion_1dTrend_Filter"
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
    
    # Load daily data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # RSI(2) on 4h close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(100).values  # Handle initial NaN
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 2)
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_4h[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI2 < 10 in daily uptrend
            if rsi[i] < 10 and ema_50_4h[i] > ema_50_4h[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: RSI2 > 90 in daily downtrend
            elif rsi[i] > 90 and ema_50_4h[i] < ema_50_4h[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI2 > 50 or trend reverses
            if rsi[i] > 50 or ema_50_4h[i] < ema_50_4h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI2 < 50 or trend reverses
            if rsi[i] < 50 or ema_50_4h[i] > ema_50_4h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: RSI(2) extreme mean reversion with daily trend filter
# - RSI(2) < 10 indicates extreme oversold conditions, ripe for mean reversion bounce
# - RSI(2) > 90 indicates extreme overbought conditions, ripe for mean reversion pullback
# - Only take signals in direction of daily trend (EMA50 slope) to avoid counter-trend trades
# - Works in both bull (buy oversold dips in uptrend) and bear (sell overbought rallies in downtrend)
# - Very few trades expected (<20/year) due to strict RSI2 extremes + trend filter
# - Position size 0.25 keeps drawdown manageable during strong trends
# - Uses 1d timeframe for trend filter, 4h for RSI2 calculation and execution timing
# - Proven concept: RSI(2) mean reversion is a well-known edge in equity markets, adapted for crypto with trend filter to avoid whipsaws
# - Low trade frequency minimizes fee drag, critical for strategy longevity