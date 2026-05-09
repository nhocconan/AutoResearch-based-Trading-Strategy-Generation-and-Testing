#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_KC_RSI20_1dTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA20 for trend filter
    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Keltner Channel (KC) parameters
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])  # First TR is undefined
    atr = np.full(n, np.nan)
    for i in range(1, n):
        if i < 11:
            atr[i] = np.nan
        else:
            if np.isnan(atr[i-1]):
                atr[i] = np.mean(tr[i-10:i+1])
            else:
                atr[i] = (atr[i-1] * 9 + tr[i]) / 10  # Wilder's smoothing
    
    # Calculate EMA20 for KC middle line
    ema20 = np.full(n, np.nan)
    ema20_val = np.nan
    for i in range(n):
        if np.isnan(ema20_val):
            if not np.isnan(close[i]):
                ema20_val = close[i]
        else:
            ema20_val = (close[i] * 2 + ema20_val * 18) / 20  # EMA(20)
        ema20[i] = ema20_val
    
    # Calculate KC bands
    kc_upper = ema20 + 2 * atr
    kc_lower = ema20 - 2 * atr
    
    # Calculate RSI(20)
    delta = np.diff(close, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(1, n):
        if i < 20:
            if not np.isnan(gain[i]) and not np.isnan(loss[i]):
                if i == 1:
                    avg_gain[i] = gain[i]
                    avg_loss[i] = loss[i]
                else:
                    avg_gain[i] = (avg_gain[i-1] * (i-1) + gain[i]) / i
                    avg_loss[i] = (avg_loss[i-1] * (i-1) + loss[i]) / i
            else:
                avg_gain[i] = avg_gain[i-1]
                avg_loss[i] = avg_loss[i-1]
        else:
            if np.isnan(avg_gain[i-1]):
                avg_gain[i] = np.mean(gain[i-19:i+1])
                avg_loss[i] = np.mean(loss[i-19:i+1])
            else:
                avg_gain[i] = (avg_gain[i-1] * 19 + gain[i]) / 20
                avg_loss[i] = (avg_loss[i-1] * 19 + loss[i]) / 20
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, np.inf)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)
    rsi = np.where(avg_gain == 0, 0, rsi)
    
    # Align daily trend to 4h
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Calculate volume confirmation (20-period average)
    vol_avg_20 = np.full(n, np.nan)
    for i in range(n):
        if i < 19:
            vol_avg_20[i] = np.nan
        else:
            vol_avg_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(ema20_1d_aligned[i]) or np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or np.isnan(rsi[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_confirmed = volume[i] > 1.5 * vol_avg_20[i]
        price = close[i]
        
        if position == 0:
            # Long entry: price touches KC lower + RSI oversold + daily uptrend
            if price <= kc_lower[i] and rsi[i] < 30 and price > ema20_1d_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
                continue
            # Short entry: price touches KC upper + RSI overbought + daily downtrend
            elif price >= kc_upper[i] and rsi[i] > 70 and price < ema20_1d_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price crosses above KC middle or trend change
            if price >= ema20[i] or price < ema20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses below KC middle or trend change
            if price <= ema20[i] or price > ema20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals