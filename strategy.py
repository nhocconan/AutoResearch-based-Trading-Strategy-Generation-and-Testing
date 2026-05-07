#!/usr/bin/env python3
"""
4h_KAMA_Direction_RSI_1dTrend_Volume
Hypothesis: KAMA (Kaufman Adaptive Moving Average) identifies trend direction, 
RSI(14) provides momentum filter, 1-day EMA50 confirms higher timeframe trend, 
and volume spikes validate the move. This combination reduces whipsaws in 
choppy markets while capturing trends in both bull and bear regimes. 
Designed for low-frequency, high-quality signals on 4H timeframe.
"""
name = "4h_KAMA_Direction_RSI_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # Parameters: ER period=10, Fast EMA=2, Slow EMA=30
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)
    # Fix volatility calculation: rolling sum of absolute changes
    volatility = pd.Series(change).rolling(window=10, min_periods=1).sum().values
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # smoothing constant
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # KAMA needs warmup, RSI needs 14, volume needs 20
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(kama[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > KAMA (uptrend) + RSI > 50 (bullish momentum) + 
            #       price > 1d EMA50 (higher timeframe uptrend) + volume confirmation
            if (close[i] > kama[i] and rsi[i] > 50 and 
                close[i] > ema_50_1d_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA (downtrend) + RSI < 50 (bearish momentum) + 
            #        price < 1d EMA50 (higher timeframe downtrend) + volume confirmation
            elif (close[i] < kama[i] and rsi[i] < 50 and 
                  close[i] < ema_50_1d_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: trend reversal signal
            if position == 1:
                if close[i] < kama[i]:  # trend turned down
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > kama[i]:  # trend turned up
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals