#!/usr/bin/env python3
# 12h_vwap_rsi_extreme_reversion_v1
# Hypothesis: Extreme RSI deviations from VWAP on 12h chart with volume confirmation and 1w trend filter.
# Long when RSI < 25 and price > VWAP (oversold bounce in uptrend), short when RSI > 75 and price < VWAP (overbought rejection in downtrend).
# Exit when RSI returns to neutral zone (40-60) or price crosses VWAP in opposite direction.
# Uses 1w EMA(50) trend filter: only take long when price > 1w EMA, short when price < 1w EMA.
# Target: 10-25 trades/year (40-100 total over 4 years) with strict entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_vwap_rsi_extreme_reversion_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate VWAP (typical price * volume) cumulative
    typical_price = (high + low + close) / 3
    vp = typical_price * volume
    cum_vp = np.zeros(n)
    cum_vol = np.zeros(n)
    cum_vp[0] = vp[0]
    cum_vol[0] = volume[0]
    for i in range(1, n):
        cum_vp[i] = cum_vp[i-1] + vp[i]
        cum_vol[i] = cum_vol[i-1] + volume[i]
    vwap = np.where(cum_vol > 0, cum_vp / cum_vol, close)
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA(50)
    close_1w = df_1w['close'].values
    ema_1w = np.zeros_like(close_1w, dtype=float)
    ema_1w[0] = close_1w[0]
    alpha = 2.0 / (50 + 1)
    for i in range(1, len(close_1w)):
        ema_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_1w[i-1]
    
    # Align 1w EMA to 12h timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(rsi[i]) or np.isnan(vwap[i]) or np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI returns to neutral zone (40-60) or price crosses below VWAP
            if rsi[i] >= 40 or close[i] < vwap[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral zone (40-60) or price crosses above VWAP
            if rsi[i] <= 60 or close[i] > vwap[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: RSI < 25 (oversold), price > VWAP, and price > 1w EMA (uptrend)
            if rsi[i] < 25 and close[i] > vwap[i] and close[i] > ema_1w_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: RSI > 75 (overbought), price < VWAP, and price < 1w EMA (downtrend)
            elif rsi[i] > 75 and close[i] < vwap[i] and close[i] < ema_1w_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals