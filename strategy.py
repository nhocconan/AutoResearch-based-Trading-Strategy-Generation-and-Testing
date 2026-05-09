#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_MultiTF_Trend_Momentum"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    """
    1h strategy using 4h trend (EMA21) and 1h momentum (RSI(14)) with volume filter.
    - Long: Price > 4h EMA21 AND RSI(14) < 30 AND volume > 1.5x avg volume (20)
    - Short: Price < 4h EMA21 AND RSI(14) > 70 AND volume > 1.5x avg volume (20)
    - Exit: Opposite condition met (price crosses 4h EMA21 or RSI reverts)
    - Session filter: 08:00-20:00 UTC only
    - Target: 15-30 trades/year on 1h timeframe (60-120 total over 4 years)
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate EMA21 on 4h close
    close_4h = df_4h['close'].values
    ema_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 21:
        ema_4h[20] = np.mean(close_4h[:21])
        for i in range(21, len(close_4h)):
            ema_4h[i] = (close_4h[i] * 2/22) + (ema_4h[i-1] * 20/22)
    
    # Align 4h EMA to 1h
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate RSI(14) on 1h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    
    # Wilder smoothing for RSI
    period = 14
    if n >= period:
        avg_gain[period-1] = np.mean(gain[1:period+1])
        avg_loss[period-1] = np.mean(loss[1:period+1])
        
        for i in range(period, n):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.full(n, np.nan)
    rsi = np.full(n, 50.0)  # default neutral
    
    for i in range(period, n):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
    
    # Volume average (20-period)
    vol_avg = np.full(n, np.nan)
    for i in range(20, n):
        vol_avg[i] = np.mean(volume[i-20:i])
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08:00-20:00 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.5x 20-period average
        vol_condition = volume[i] > vol_avg[i] * 1.5
        
        if position == 0:
            # Long: Price > 4h EMA21 AND RSI < 30 (oversold) AND volume spike
            if (close[i] > ema_4h_aligned[i] and rsi[i] < 30 and vol_condition):
                signals[i] = 0.20
                position = 1
            # Short: Price < 4h EMA21 AND RSI > 70 (overbought) AND volume spike
            elif (close[i] < ema_4h_aligned[i] and rsi[i] > 70 and vol_condition):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Price < 4h EMA21 OR RSI > 50 (momentum fade)
            if close[i] < ema_4h_aligned[i] or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Price > 4h EMA21 OR RSI < 50 (momentum fade)
            if close[i] > ema_4h_aligned[i] or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals