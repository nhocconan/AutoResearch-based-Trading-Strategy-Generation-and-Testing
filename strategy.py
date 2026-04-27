#!/usr/bin/env python3
"""
Hypothesis: 1-hour RSI(14) mean reversion with 4-hour trend filter and volume spike filter.
- Long when RSI < 30 (oversold) + price > 4h EMA50 (uptrend) + volume spike (>1.5x avg)
- Short when RSI > 70 (overbought) + price < 4h EMA50 (downtrend) + volume spike (>1.5x avg)
- Exit when RSI crosses back to neutral (40 for longs, 60 for shorts) or trend reverses
- Volume confirmation reduces false signals in low volatility
- Uses 4h for trend direction (fewer signals), 1h for precise entry timing
- Target: 15-30 trades/year (60-120 over 4 years) to minimize fee drag
- Works in bull/bear via trend filter - only trades with higher timeframe trend
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
    
    # Calculate 1h RSI(14) - use Wilder's smoothing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing: alpha = 1/period
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[13] = np.mean(gain[1:14])  # First average
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for RSI and EMA
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: RSI oversold + above 4h EMA50 + volume spike
            if (rsi[i] < 30 and close[i] > ema50_4h_aligned[i] and volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: RSI overbought + below 4h EMA50 + volume spike
            elif (rsi[i] > 70 and close[i] < ema50_4h_aligned[i] and volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI crosses above 40 (mean reversion complete) OR below 4h EMA50 (trend change)
            if (rsi[i] > 40 or close[i] < ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RSI crosses below 60 (mean reversion complete) OR above 4h EMA50 (trend change)
            if (rsi[i] < 60 or close[i] > ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI14_MeanRev_4hEMA50_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0