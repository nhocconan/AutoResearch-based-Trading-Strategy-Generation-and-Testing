#!/usr/bin/env python3
# 6h_Liquidity_Grab_Reversal_12hTrend_Volume
# Hypothesis: Price often spikes to liquidity zones (prior 12h high/low) then reverses.
# We fade these spikes when: 1) price touches 12h high/low, 2) RSI shows divergence,
# 3) volume dries up on the spike, and 4) 12h trend provides bias. Works in ranging
# and trending markets by fading exhaustion moves.

name = "6h_Liquidity_Grab_Reversal_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for trend and liquidity levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h high/low for liquidity zones
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h EMA50 trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_12h_up = close_12h > ema50_12h
    trend_12h_down = close_12h < ema50_12h
    
    # Align 12h data to 6h
    high_12h_aligned = align_htf_to_ltf(prices, df_12h, high_12h)
    low_12h_aligned = align_htf_to_ltf(prices, df_12h, low_12h)
    trend_12h_up_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_up.astype(float))
    trend_12h_down_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_down.astype(float))
    
    # RSI(14) for momentum/divergence
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Volume filter: current < 0.5 * 20-period average (drying up on spike)
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(high_12h_aligned[i]) or np.isnan(low_12h_aligned[i]) or
            np.isnan(trend_12h_up_aligned[i]) or np.isnan(trend_12h_down_aligned[i]) or
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] < (0.5 * vol_ma[i]) if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long setup: price touches 12h low, RSI bullish divergence, volume drying
            if (low[i] <= low_12h_aligned[i] and 
                rsi[i] < 30 and  # oversold
                vol_condition and
                trend_12h_up_aligned[i] > 0.5):  # 12h uptrend bias
                signals[i] = 0.25
                position = 1
            # Short setup: price touches 12h high, RSI bearish divergence, volume drying
            elif (high[i] >= high_12h_aligned[i] and 
                  rsi[i] > 70 and  # overbought
                  vol_condition and
                  trend_12h_down_aligned[i] > 0.5):  # 12h downtrend bias
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: RSI normalizes or stop at 12h high
            if (rsi[i] > 50 or 
                high[i] >= high_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: RSI normalizes or stop at 12h low
            if (rsi[i] < 50 or 
                low[i] <= low_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals