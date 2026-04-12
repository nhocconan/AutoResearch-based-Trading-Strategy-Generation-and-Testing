#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_rsi_volatility_filter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Hour index for session filter (08-20 UTC)
    hours = prices.index.hour
    
    # Daily RSI(14) for trend bias
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # RSI(14) on daily
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14 = rsi_14.values
    
    # Align daily RSI to 1h
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    # 1h volatility filter: ATR(24) > 20-period SMA of ATR
    tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=24, min_periods=24).mean()
    atr_ma = atr.rolling(window=20, min_periods=20).mean()
    vol_filter = atr > atr_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # warmup for indicators
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Skip if not ready
        if (np.isnan(rsi_14_aligned[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Long: daily RSI < 40 (oversold) + volatility filter
        # Short: daily RSI > 60 (overbought) + volatility filter
        long_signal = rsi_14_aligned[i] < 40 and vol_filter[i]
        short_signal = rsi_14_aligned[i] > 60 and vol_filter[i]
        
        # Exit when RSI returns to neutral zone (40-60)
        exit_long = rsi_14_aligned[i] >= 50
        exit_short = rsi_14_aligned[i] <= 50
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.20
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals