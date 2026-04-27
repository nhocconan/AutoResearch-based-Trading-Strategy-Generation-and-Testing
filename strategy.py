#!/usr/bin/env python3
"""
12h_RSI20_Pullback_1dTrend_Volume
Hypothesis: RSI(20) < 30 identifies oversold pullbacks in uptrend, RSI(20) > 70 identifies overbought bounces in downtrend, filtered by 1d EMA50 trend and volume spike. Works in bull markets via long pullbacks and in bear markets via short bounces. Targets ~15-25 trades/year on 12h to minimize fee drag.
"""

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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # RSI(20) on 12h data
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (alpha = 1/period)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[19] = np.mean(gain[1:20])  # first average
    avg_loss[19] = np.mean(loss[1:20])
    
    for i in range(20, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 19 + gain[i]) / 20
        avg_loss[i] = (avg_loss[i-1] * 19 + loss[i]) / 20
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 1.8 * 24-period average (2 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for RSI and volume MA
    start_idx = max(50, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        rsi_val = rsi[i]
        ema_trend = ema50_1d_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: RSI < 30 (oversold) with uptrend and volume spike
            if rsi_val < 30 and vol_spike_val and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: RSI > 70 (overbought) with downtrend and volume spike
            elif rsi_val > 70 and vol_spike_val and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI > 50 (mean reversion) or trend turns down
            if rsi_val > 50 or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI < 50 (mean reversion) or trend turns up
            if rsi_val < 50 or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_RSI20_Pullback_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0