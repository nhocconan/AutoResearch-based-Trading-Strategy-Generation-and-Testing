#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h trend filter and volume spike. Long when price > 4h EMA50 + 1h RSI > 55 + volume spike > 2x avg; short when price < 4h EMA50 + 1h RSI < 45 + volume spike > 2x avg. Exit on RSI mean reversion (50). Use 08-20 UTC session filter to avoid low-volume hours. Target 100-200 total trades over 4 years.

name = "1h_momentum_4h_ema50_rsi_vol_spike_v1"
timeframe = "1h"
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
    
    # 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_4h_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_50)
    
    # 1h RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_ema = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    loss_ema = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = gain_ema / (loss_ema + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_4h_50_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: RSI mean reversion to 50
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: RSI mean reversion to 50
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: momentum with trend filter and volume spike
            # Long: price > 4h EMA50 + RSI > 55 + volume spike
            if (close[i] > ema_4h_50_aligned[i] and 
                rsi[i] > 55 and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: price < 4h EMA50 + RSI < 45 + volume spike
            elif (close[i] < ema_4h_50_aligned[i] and 
                  rsi[i] < 45 and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals