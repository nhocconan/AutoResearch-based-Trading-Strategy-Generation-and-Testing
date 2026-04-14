#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h RSI extreme + 1d EMA trend filter with volume confirmation.
# RSI < 30 for long, RSI > 70 for short on 4h timeframe captures mean reversion in both bull and bear markets.
# 1d EMA(50) filter ensures trades align with daily trend (long when above, short when below).
# Volume > 1.5x 20-period average confirms momentum.
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag.
# Entry only during active session (08-20 UTC) to reduce noise.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE for RSI
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Calculate RSI on 4h data
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    rsi_period = 14
    avg_gain = pd.Series(gain).ewm(span=rsi_period, adjust=False, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, adjust=False, min_periods=rsi_period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Load 1d data ONCE for EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA on 1d data
    close_1d = df_1d['close'].values
    ema_period = 50
    ema = pd.Series(close_1d).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    
    # Align indicators to 1h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_4h, rsi)
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = max(14, 50, 20)
    
    for i in range(start, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any critical data is NaN
        if (np.isnan(rsi_aligned[i]) or 
            np.isnan(ema_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: RSI oversold (<30) AND price above daily EMA (uptrend) AND volume
            if (rsi_aligned[i] < 30 and 
                close[i] > ema_aligned[i] and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: RSI overbought (>70) AND price below daily EMA (downtrend) AND volume
            elif (rsi_aligned[i] > 70 and 
                  close[i] < ema_aligned[i] and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral (50) or price crosses below EMA
            if (rsi_aligned[i] >= 50 or 
                close[i] < ema_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI returns to neutral (50) or price crosses above EMA
            if (rsi_aligned[i] <= 50 or 
                close[i] > ema_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4hRSI_1dEMA_VolumeFilter_v1"
timeframe = "1h"
leverage = 1.0