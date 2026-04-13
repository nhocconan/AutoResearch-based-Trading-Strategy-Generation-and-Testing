#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h 12-hour KAMA trend with 1d RSI filter and volume confirmation
    # Long: KAMA rising + RSI > 50 + volume > 1.5x 20-period average
    # Short: KAMA falling + RSI < 50 + volume > 1.5x 20-period average
    # Uses discrete sizing (0.25) to minimize fee drag and ATR-based stoploss
    # Target: 20-50 trades/year to stay within 4h optimal range (80-200 total over 4 years)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d RSI(14)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i < 14:
            if i == 0:
                avg_gain[i] = np.nan
                avg_loss[i] = np.nan
            else:
                avg_gain[i] = np.mean(gain[1:i+1]) if i >= 1 else 0
                avg_loss[i] = np.mean(loss[1:i+1]) if i >= 1 else 0
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = np.insert(rsi_1d, 0, np.nan)  # align with close_1d index
    
    # Align 1d RSI to 4h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate KAMA(10,2,30) on 4h close
    # ER = |close - close[10]| / sum|close[i] - close[i-1]| for i=1 to 10
    change = np.abs(close - np.roll(close, 10))
    change[0:10] = np.nan  # not enough data
    volatility = np.zeros(n)
    for i in range(1, n):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    volatility = np.abs(np.diff(volatility, n=10))  # 10-period sum of absolute changes
    volatility = np.insert(volatility, 0, [np.nan]*10)  # align indices
    er = np.divide(change, volatility, out=np.full_like(change, np.nan), where=volatility!=0)
    # Smooth constant: SC = [ER * (fastest - slowest) + slowest]^2
    fastest = 2/(2+1)  # EMA(2)
    slowest = 2/(30+1) # EMA(30)
    sc = (er * (fastest - slowest) + slowest) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate volume average for confirmation (using 4h data)
    vol_avg_20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_avg_20[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else 0
        else:
            vol_avg_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    # Calculate ATR using true range approximation for 4h timeframe
    atr_4h = np.zeros(n)
    for i in range(1, n):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        if i < 14:
            atr_4h[i] = tr  # Simple average for warmup
        else:
            atr_4h[i] = 0.93 * atr_4h[i-1] + 0.07 * tr  # Wilder's smoothing
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(kama[i-1]) or 
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(vol_avg_20[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction: rising/falling
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        # RSI filter: >50 for long, <50 for short
        rsi_long = rsi_1d_aligned[i] > 50
        rsi_short = rsi_1d_aligned[i] < 50
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_avg_20[i]
        
        # Entry conditions
        enter_long = kama_rising and rsi_long and volume_confirmed
        enter_short = kama_falling and rsi_short and volume_confirmed
        
        # Stoploss: 2x ATR below/above entry
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - 2.0 * atr_4h[i]
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + 2.0 * atr_4h[i]
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            elif position == -1:
                signals[i] = -position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            else:
                signals[i] = 0.0
                entry_price[i] = np.nan
    
    return signals

name = "4h_12h_kama_rsi_volume_v1"
timeframe = "4h"
leverage = 1.0