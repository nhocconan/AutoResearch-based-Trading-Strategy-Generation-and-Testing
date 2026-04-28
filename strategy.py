#!/usr/bin/env python3
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
    
    # Get daily data for ATR-based volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on daily
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr14 = np.zeros_like(close_1d)
    atr14[14] = np.mean(tr[1:15])
    for i in range(15, len(tr)):
        atr14[i] = (atr14[i-1] * 13 + tr[i]) / 14
    
    # ATR-based volatility filter: only trade when volatility is elevated
    atr_ma = np.zeros_like(atr14)
    for i in range(len(atr_ma)):
        if i < 10:
            atr_ma[i] = np.nan
        else:
            atr_ma[i] = np.mean(atr14[max(0, i-9):i+1])
    vol_filter = atr14 > atr_ma  # Trade when current ATR > 10-period MA of ATR
    
    # Align volatility filter to 4h
    vol_filter_aligned = align_htf_to_ltf(prices, df_1d, vol_filter.astype(float))
    
    # Calculate 4-period RSI on 4h for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[3] = np.mean(gain[1:4])
    avg_loss[3] = np.mean(loss[1:4])
    
    for i in range(4, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 3 + gain[i]) / 4
        avg_loss[i] = (avg_loss[i-1] * 3 + loss[i]) / 4
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 20-period SMA for trend filter
    sma20 = np.zeros_like(close)
    for i in range(len(sma20)):
        if i < 19:
            sma20[i] = np.nan
        else:
            sma20[i] = np.mean(close[i-19:i+1])
    
    # Mean reversion signals: RSI extremes with trend filter
    rsi_oversold = rsi < 30
    rsi_overbought = rsi > 70
    uptrend = close > sma20
    downtrend = close < sma20
    
    long_signal = rsi_oversold & uptrend
    short_signal = rsi_overbought & downtrend
    
    # Combine with volatility filter and session
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or 
            np.isnan(sma20[i]) or
            np.isnan(vol_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when volatility is elevated
        if not vol_filter_aligned[i]:
            signals[i] = 0.0
            continue
        
        # Entry conditions
        long_entry = long_signal[i]
        short_entry = short_signal[i]
        
        # Exit conditions: RSI returns to neutral zone
        long_exit = rsi[i] >= 50
        short_exit = rsi[i] <= 50
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_RSI_MeanReversion_VolatilityFilter"
timeframe = "4h"
leverage = 1.0