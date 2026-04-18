#!/usr/bin/env python3
"""
12h_1d_KAMA_Direction_RSI_Threshold
Hypothesis: Uses 12h KAMA to determine trend direction, enters when RSI(14) shows weakness in a trending market (pullback to mean).
Combines trend-following with mean-reversion entries to capture swings in both bull and bear markets.
Filters: requires volume confirmation and avoids choppy markets using ADX(14) < 25.
Target: 15-25 trades/year by using strong trend filter and precise entry conditions.
"""

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
    
    # Get 1d data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align RSI to 12h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Get 12h KAMA for trend direction
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate Efficiency Ratio for KAMA
    change = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    volatility = np.sum(np.abs(np.diff(close_12h, prepend=close_12h[0])), axis=0) if False else None  # placeholder
    
    # Proper ER calculation
    er = np.full(len(close_12h), np.nan)
    for i in range(10, len(close_12h)):
        direction = np.abs(close_12h[i] - close_12h[i-9])
        volatility = np.sum(np.abs(np.diff(close_12h[i-9:i+1])))
        if volatility > 0:
            er[i] = direction / volatility
        else:
            er[i] = 0
    
    # Smoothing constants
    sc = (er * 0.6 + 0.064) ** 2  # where 0.6 = 2/(2+1), 0.064 = 2/(30+1)
    
    # Calculate KAMA
    kama = np.full(len(close_12h), np.nan)
    kama[9] = close_12h[9]  # seed
    for i in range(10, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    
    # Align KAMA to 12h timeframe (though data is already 12h, align for consistency)
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    # ADX(14) for choppy market filter on 12h data
    # Calculate +DI, -DI, and DX
    plus_dm = np.zeros(len(close_12h))
    minus_dm = np.zeros(len(close_12h))
    tr = np.zeros(len(close_12h))
    
    for i in range(1, len(close_12h)):
        high_diff = df_12h['high'].iloc[i] - df_12h['high'].iloc[i-1]
        low_diff = df_12h['low'].iloc[i-1] - df_12h['low'].iloc[i]
        
        plus_dm[i] = max(high_diff, 0) if high_diff > low_diff else 0
        minus_dm[i] = max(low_diff, 0) if low_diff > high_diff else 0
        
        tr[i] = max(
            df_12h['high'].iloc[i] - df_12h['low'].iloc[i],
            np.abs(df_12h['high'].iloc[i] - df_12h['close'].iloc[i-1]),
            np.abs(df_12h['low'].iloc[i] - df_12h['close'].iloc[i-1])
        )
    
    # Smooth TR, +DM, -DM
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / (atr + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(kama_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Avoid choppy markets: only trade when ADX < 25 (ranging market)
        if adx_aligned[i] >= 25:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price pulls back to KAMA in uptrend with RSI showing exhaustion
            if (close[i] > kama_aligned[i] and  # uptrend
                close[i] <= kama_aligned[i] * 1.02 and  # near KAMA
                rsi_1d_aligned[i] < 30 and  # oversold
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price pulls back to KAMA in downtrend with RSI showing exhaustion
            elif (close[i] < kama_aligned[i] and  # downtrend
                  close[i] >= kama_aligned[i] * 0.98 and  # near KAMA
                  rsi_1d_aligned[i] > 70 and  # overbought
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: RSI shows strength or price moves significantly above KAMA
            if (rsi_1d_aligned[i] > 50 or 
                close[i] > kama_aligned[i] * 1.05):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI shows weakness or price moves significantly below KAMA
            if (rsi_1d_aligned[i] < 50 or 
                close[i] < kama_aligned[i] * 0.95):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_KAMA_Direction_RSI_Threshold"
timeframe = "12h"
leverage = 1.0