#!/usr/bin/env python3
"""
12h_1d_RSI_Overbought_Oversold_v1
Hypothesis: Trade daily RSI extremes (overbought >70, oversold <30) with volume > 1.5x 20-period average on 12h timeframe.
Use 12h EMA50 for trend filter: long only in uptrend, short only in downtrend.
Exit on RSI returning to neutral zone (40-60) or trend reversal.
Designed for low trade frequency (~15-30/year) with high conviction.
Works in bull markets (oversold bounces in uptrend) and bear markets (overbought reversals in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_RSI_Overbought_Oversold_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA FOR RSI CALCULATION ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate RSI(14) on daily closes
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align RSI to 12h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === TREND FILTER: 12h EMA50 ===
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === VOLUME FILTER ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(ema50[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine trend
        uptrend = close[i] > ema50[i]
        downtrend = close[i] < ema50[i]
        
        # Volume strength
        strong_volume = volume[i] > (vol_ma[i] * 1.5)
        
        # RSI levels
        rsi = rsi_1d_aligned[i]
        rsi_overbought = rsi > 70
        rsi_oversold = rsi < 30
        rsi_neutral = (rsi >= 40) & (rsi <= 60)
        
        # Long: RSI oversold with strong volume in uptrend
        long_signal = rsi_oversold and uptrend and strong_volume
        
        # Short: RSI overbought with strong volume in downtrend
        short_signal = rsi_overbought and downtrend and strong_volume
        
        # Exit: RSI returns to neutral or trend reversal
        exit_long = position == 1 and (rsi_neutral or not uptrend)
        exit_short = position == -1 and (rsi_neutral or not downtrend)
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals