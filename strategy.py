#!/usr/bin/env python3
"""
1h_Momentum_With_4hTrend_And_1dVolume
Hypothesis: Uses 4h EMA for trend direction and 1d volume spike for momentum confirmation, with 1h for precise entry timing. Designed for low trade frequency (~20-30 trades/year) by requiring confluence of trend, volume, and price action. Works in bull markets via trend following and bear markets via mean reversion at extreme volume spikes.
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
    
    # 4h EMA for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    ema40_4h = pd.Series(df_4h['close']).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema40_4h_aligned = align_htf_to_ltf(prices, df_4h, ema40_4h)
    
    # 1d volume spike (volume > 2.5x 20-day average)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    vol_avg_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = df_1d['volume'].values > (2.5 * vol_avg_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # 1h RSI for overbought/oversold conditions
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # Position size: 20% of capital
    
    # Warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema40_4h_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        ema40 = ema40_4h_aligned[i]
        vol_spike = vol_spike_1d_aligned[i] > 0.5  # boolean
        rsi_val = rsi[i]
        
        if position == 0:
            # Long: uptrend (price > EMA40), volume spike, RSI not overbought
            if close[i] > ema40 and vol_spike and rsi_val < 70:
                signals[i] = size
                position = 1
            # Short: downtrend (price < EMA40), volume spike, RSI not oversold
            elif close[i] < ema40 and vol_spike and rsi_val > 30:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: trend reversal or overbought
            if close[i] < ema40 or rsi_val > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: trend reversal or oversold
            if close[i] > ema40 or rsi_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Momentum_With_4hTrend_And_1dVolume"
timeframe = "1h"
leverage = 1.0