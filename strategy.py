#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h trend filter and volume confirmation.
# Uses 4h EMA for trend direction, 1h RSI for momentum entry, and volume spike for confirmation.
# Designed for low-frequency trading (15-30 trades/year) to minimize fee drag.
# Works in bull markets via trend following and bear markets via mean reversion at extremes.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h RSI (14-period) for momentum signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # 50-period EMA on 4h close for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume filter: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: uptrend + oversold bounce + volume
        if (close[i] > ema50_4h_aligned[i] and  # 4h uptrend
            rsi[i] < 30 and  # Oversold
            volume_filter[i]):  # Volume confirmation
            signals[i] = 0.20
        
        # Short conditions: downtrend + overbought rejection + volume
        elif (close[i] < ema50_4h_aligned[i] and  # 4h downtrend
              rsi[i] > 70 and  # Overbought
              volume_filter[i]):  # Volume confirmation
            signals[i] = -0.20
    
    return signals

name = "1h_RSI14_4hEMA50_VolumeFilter"
timeframe = "1h"
leverage = 1.0