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
    
    # Load daily data (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily pivot points (classic)
    if len(high_1d) < 1:
        return np.zeros(n)
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 20-period EMA for trend filter (daily)
    if len(close_1d) < 20:
        return np.zeros(n)
    
    ema20_1d = np.full_like(close_1d, np.nan)
    ema20_1d[19] = np.mean(close_1d[:20])
    for i in range(20, len(close_1d)):
        ema20_1d[i] = close_1d[i] * 0.0952 + ema20_1d[i-1] * 0.9048  # alpha = 2/(20+1)
    
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Calculate 14-day RSI for momentum (daily)
    if len(close_1d) < 14:
        return np.zeros(n)
    
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= 14:
        avg_gain[13] = np.mean(gain[1:14])
        avg_loss[13] = np.mean(loss[1:14])
        for i in range(14, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.full_like(close_1d, np.nan)
    rsi14 = np.full_like(close_1d, np.nan)
    for i in range(13, len(close_1d)):
        if avg_loss[i] > 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi14[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi14[i] = 100 if avg_gain[i] > 0 else 0
    
    rsi14_aligned = align_htf_to_ltf(prices, df_1d, rsi14)
    
    # Calculate 14-day ATR for volatility filter (daily)
    if len(high_1d) < 14 or len(low_1d) < 14 or len(close_1d) < 14:
        return np.zeros(n)
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR
    
    atr14 = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 14:
        atr14[13] = np.mean(tr[1:14])
        for i in range(14, len(close_1d)):
            atr14[i] = (atr14[i-1] * 13 + tr[i]) / 14
    
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    
    # Volume moving average (20 periods) for volume spike detection
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema20_1d_aligned[i]) or 
            np.isnan(rsi14_aligned[i]) or 
            np.isnan(atr14_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume ratio: current volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        # Long conditions: Price above S1, above EMA20, RSI > 55, volume surge (>2x), ATR positive
        long_condition = (close[i] > s1_aligned[i] and
                         close[i] > ema20_1d_aligned[i] and
                         rsi14_aligned[i] > 55 and
                         volume_ratio > 2.0 and
                         atr14_aligned[i] > 0)
        
        # Short conditions: Price below R1, below EMA20, RSI < 45, volume surge (>2x), ATR positive
        short_condition = (close[i] < r1_aligned[i] and
                          close[i] < ema20_1d_aligned[i] and
                          rsi14_aligned[i] < 45 and
                          volume_ratio > 2.0 and
                          atr14_aligned[i] > 0)
        
        if position == 0:
            if long_condition:
                position = 1
                signals[i] = position_size
            elif short_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price below S1 OR RSI < 40
            if (close[i] < s1_aligned[i] or 
                rsi14_aligned[i] < 40):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price above R1 OR RSI > 60
            if (close[i] > r1_aligned[i] or 
                rsi14_aligned[i] > 60):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_1d_Pivot_R1S1_EMA20_RSI14_Volume_Session"
timeframe = "1h"
leverage = 1.0