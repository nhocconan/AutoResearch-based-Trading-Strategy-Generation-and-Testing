#!/usr/bin/env python3
"""
1h RSI Mean Reversion with 4h Trend Filter and Volume Spike
Designed for low trade frequency (target: 15-37 trades/year) by combining:
- RSI(14) extremes for mean reversion entries
- 4h EMA trend filter to align with higher timeframe direction
- Volume spike confirmation to filter noise
- Session filter (08-20 UTC) to reduce noise
Works in both bull and bear markets by only taking mean reversion in direction of 4h trend.
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA34 for trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # RSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike detection (2x 4-period average)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 34  # need enough history for EMA and RSI
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        rsi_val = rsi[i]
        ema_trend = ema_34_4h_aligned[i]
        price = close[i]
        
        if position == 0:
            # Long: RSI oversold (<30) with volume spike and price above 4h EMA (uptrend)
            if (rsi_val < 30 and 
                volume_spike[i] and 
                price > ema_trend):
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought (>70) with volume spike and price below 4h EMA (downtrend)
            elif (rsi_val > 70 and 
                  volume_spike[i] and 
                  price < ema_trend):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long position: hold until RSI returns to neutral or trend changes
            signals[i] = 0.20
            if rsi_val >= 50 or price < ema_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position: hold until RSI returns to neutral or trend changes
            signals[i] = -0.20
            if rsi_val <= 50 or price > ema_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_RSI_MeanReversion_4hEMA34_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0