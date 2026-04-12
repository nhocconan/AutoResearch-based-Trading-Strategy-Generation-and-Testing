#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Precompute hour filter for 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA(200) for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 1d RSI(14) for mean reversion signals
    delta = np.diff(close_1d, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full(len(delta), np.nan)
    avg_loss = np.full(len(delta), np.nan)
    for i in range(14, len(delta)):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align 1d indicators to daily timeframe (1:1 mapping)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 200 EMA
        uptrend = close[i] > ema_200_1d_aligned[i]
        downtrend = close[i] < ema_200_1d_aligned[i]
        
        # Mean reversion signals: RSI extremes
        rsi_oversold = rsi_1d_aligned[i] < 30
        rsi_overbought = rsi_1d_aligned[i] > 70
        
        # Entry conditions: 
        # Long: uptrend + RSI oversold (buy dips in uptrend)
        # Short: downtrend + RSI overbought (sell rallies in downtrend)
        long_entry = uptrend and rsi_oversold
        short_entry = downtrend and rsi_overbought
        
        # Exit conditions: opposite trend or RSI mean reversion
        long_exit = (not uptrend) or (rsi_1d_aligned[i] > 50)
        short_exit = (not downtrend) or (rsi_1d_aligned[i] < 50)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_ema200_rsi_mean_reversion_v1"
timeframe = "1d"
leverage = 1.0