#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h trend filter and volume confirmation. 
# Uses 4h EMA20 for trend direction (bullish if close > EMA20, bearish if close < EMA20) 
# and 1h RSI(14) for momentum entries (long when RSI < 30 and rising, short when RSI > 70 and falling).
# Volume confirmation requires current volume > 1.5x 20 EMA volume. 
# Session filter (08-20 UTC) reduces noise. 
# Works in bull markets (buying dips in uptrend) and bear markets (selling rallies in downtrend).
# Target: 15-37 trades/year to avoid fee drag.

name = "1h_RSI_Momentum_4hEMA20_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h EMA20 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume confirmation: volume > 1.5x 20 EMA volume
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ema20[i]) or 
            np.isnan(rsi[i-1])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 30 and rising (RSI > previous RSI) + 4h EMA20 up + volume
            if (rsi[i] < 30 and rsi[i] > rsi[i-1] and close[i] > ema_4h_aligned[i] and vol_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Short: RSI > 70 and falling (RSI < previous RSI) + 4h EMA20 down + volume
            elif (rsi[i] > 70 and rsi[i] < rsi[i-1] and close[i] < ema_4h_aligned[i] and vol_confirm[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: RSI > 70 or trend change
            if rsi[i] > 70 or close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI < 30 or trend change
            if rsi[i] < 30 or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals