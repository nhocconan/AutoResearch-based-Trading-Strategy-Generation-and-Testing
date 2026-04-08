#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_chop_v1"
timeframe = "1d"
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate KAMA on daily close
    # Efficiency ratio
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.zeros_like(change)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    # Pad ER array to match close length
    er_full = np.full_like(close, np.nan, dtype=float)
    er_full[10:] = er
    # Smoothing constants
    sc = (er_full * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Initialize
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14) on daily close
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad RSI array
    rsi_full = np.full_like(close, np.nan)
    rsi_full[14:] = rsi
    
    # Calculate Choppiness Index on daily data
    atr = np.zeros_like(close)
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.insert(tr, 0, 0)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_hl = highest_high - lowest_low
    chop = np.zeros_like(close)
    mask = (range_hl != 0) & (atr != 0) & (~np.isnan(atr))
    chop[mask] = 100 * np.log10(np.sum(tr[np.arange(len(tr))[:, None] & np.arange(14)[None, :] >= np.arange(14)[:, None]], axis=1) / 
                                (atr * 14)) / np.log10(14)
    chop = np.where(chop > 0, chop, 50)  # Default to middle if invalid
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup period
    start_idx = max(34, 14, 10) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data not available
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(rsi_full[i]) or np.isnan(chop[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below KAMA (trend change)
            if close[i] < kama[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price crosses above KAMA (trend change)
            if close[i] > kama[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price above KAMA, RSI < 40, Chop > 61.8 (trend in chop)
            if (close[i] > kama[i] and rsi_full[i] < 40 and chop[i] > 61.8):
                position = 1
                signals[i] = 0.25
            # Short entry: price below KAMA, RSI > 60, Chop > 61.8 (trend in chop)
            elif (close[i] < kama[i] and rsi_full[i] > 60 and chop[i] > 61.8):
                position = -1
                signals[i] = -0.25
    
    return signals