#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 12h trend filter and volume confirmation
# Uses 12h EMA(34) for trend direction, 6h RSI(14) for momentum exhaustion,
# and volume spike (>1.5x 20-period average) for entry confirmation.
# Designed for low trade frequency (target: 15-35 trades/year) to minimize fee drag.
# Works in bull markets via trend continuation and in bear markets via mean reversion at extremes.

name = "6h_ema34_rsi_volume_spike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h EMA(34) for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 35:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 6h RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike detection (>1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 0)
    vol_spike = vol_ratio > 1.5
    
    signals = np.zeros(n)
    
    for i in range(34, n):
        # Skip if required data not available
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend direction from 12h EMA
        bullish_trend = close[i] > ema_12h_aligned[i]
        bearish_trend = close[i] < ema_12h_aligned[i]
        
        # RSI conditions for momentum exhaustion
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Long conditions: bullish trend + RSI oversold + volume spike
        if bullish_trend and rsi_oversold and vol_spike[i]:
            signals[i] = 0.25
        # Short conditions: bearish trend + RSI overbought + volume spike
        elif bearish_trend and rsi_overbought and vol_spike[i]:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals