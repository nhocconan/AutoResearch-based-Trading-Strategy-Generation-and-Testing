#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get higher timeframe data
    df_1d = get_htf_data(prices, '1d')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 1d EMA200 for trend
    close_1d = pd.Series(df_1d['close'])
    ema_200_1d = close_1d.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 4h MACD (12,26,9)
    close_4h = pd.Series(df_4h['close'])
    ema_12 = close_4h.ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_26 = close_4h.ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema_12 - ema_26
    macd_signal = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_line_aligned = align_htf_to_ltf(prices, df_4h, macd_line)
    macd_signal_aligned = align_htf_to_ltf(prices, df_4h, macd_signal)
    
    # Calculate 1h ATR (14) for stop loss
    high = pd.Series(prices['high'])
    low = pd.Series(prices['low'])
    close = pd.Series(prices['close'])
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1h = tr.rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: 20-period average
    volume = pd.Series(prices['volume'])
    vol_ma = volume.rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08:00 to 20:00 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(200, n):  # Start after warmup
        # Skip if NaN in critical values
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(macd_line_aligned[i]) or 
            np.isnan(macd_signal_aligned[i]) or np.isnan(atr_1h[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: price > EMA200, MACD bullish, volume spike
            if (price > ema_200_1d_aligned[i] and 
                macd_line_aligned[i] > macd_signal_aligned[i] and 
                vol > 1.5 * vol_ma[i]):
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short entry: price < EMA200, MACD bearish, volume spike
            elif (price < ema_200_1d_aligned[i] and 
                  macd_line_aligned[i] < macd_signal_aligned[i] and 
                  vol > 1.5 * vol_ma[i]):
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: stop loss (2*ATR) or trend/MACD reversal
            if (price <= entry_price - 2.0 * atr_1h[i] or
                price < ema_200_1d_aligned[i] or
                macd_line_aligned[i] < macd_signal_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: stop loss (2*ATR) or trend/MACD reversal
            if (price >= entry_price + 2.0 * atr_1h[i] or
                price > ema_200_1d_aligned[i] or
                macd_line_aligned[i] > macd_signal_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA200_MACD_Volume_Session"
timeframe = "1h"
leverage = 1.0