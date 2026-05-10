#!/usr/bin/env python3
# 6h_12h_1d_Momentum_Reversal_Scalp
# Hypothesis: In 6h timeframe, short-term momentum reversals provide edge when aligned with 12h/1d trend.
# Uses RSI(2) for extreme short-term momentum + 12h/1d EMA trend filter + volume confirmation.
# Works in bull/bear by requiring trend alignment. Targets 15-35 trades/year per symbol.

name = "6h_12h_1d_Momentum_Reversal_Scalp"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for trend context and 12h data for intermediate trend
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    if len(df_1d) < 2 or len(df_12h) < 2:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(2) for short-term momentum extremes
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, min_periods=2, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/2, min_periods=2, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
    
    # Daily EMA for higher timeframe trend
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Volume confirmation (20-period for 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough history for RSI and EMAs
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(rsi_values[i]) or
            np.isnan(ema_12h_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend alignment: both 12h and 1d must agree
        uptrend_12h = close_12h_aligned[i] > ema_12h_aligned[i]
        uptrend_1d = close_1d_aligned[i] > ema_1d_aligned[i]
        downtrend_12h = close_12h_aligned[i] < ema_12h_aligned[i]
        downtrend_1d = close_1d_aligned[i] < ema_1d_aligned[i]
        
        uptrend = uptrend_12h and uptrend_1d
        downtrend = downtrend_12h and downtrend_1d
        
        # Volume confirmation (1.5x average)
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: RSI oversold (<10) in uptrend with volume
            if rsi_values[i] < 10 and uptrend and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>90) in downtrend with volume
            elif rsi_values[i] > 90 and downtrend and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Long exit: RSI returns to neutral or trend fails
                if rsi_values[i] > 50 or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: RSI returns to neutral or trend fails
                if rsi_values[i] < 50 or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals