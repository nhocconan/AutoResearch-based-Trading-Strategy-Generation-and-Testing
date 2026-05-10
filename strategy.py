#!/usr/bin/env python3
# 6h_Vol_Spike_Reversal_1dTrend
# Hypothesis: Enter short-term reversals after volume spikes in the direction of the 1d trend.
# Long when: volume spike + price > 1d EMA50 + RSI(14) < 40
# Short when: volume spike + price < 1d EMA50 + RSI(14) > 60
# Exit when RSI crosses 50 or opposite volume spike occurs.
# Uses volume spike to catch exhaustion moves and 1d trend to filter direction.
# Targets 15-30 trades per year on 6h timeframe with position size 0.25.

name = "6h_Vol_Spike_Reversal_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate RSI(14) on 6h closes
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(volume_spike[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        rsi_neutral = 40 <= rsi[i] <= 60
        
        if position == 0:
            # Long entry: volume spike + uptrend + RSI oversold
            if (volume_spike[i] and 
                price_above_ema and 
                rsi_oversold):
                signals[i] = 0.25
                position = 1
            # Short entry: volume spike + downtrend + RSI overbought
            elif (volume_spike[i] and 
                  price_below_ema and 
                  rsi_overbought):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI crosses 50 or opposite volume spike in downtrend
            if (rsi[i] >= 50 or 
                (volume_spike[i] and price_below_ema)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI crosses 50 or opposite volume spike in uptrend
            if (rsi[i] <= 50 or 
                (volume_spike[i] and price_above_ema)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals