#!/usr/bin/env python3

"""
Hypothesis: 1-day RSI with 1-week trend filter and volume confirmation. 
RSI(14) identifies overbought/oversold conditions, while the 1-week EMA(34) 
determines the primary trend direction. Volume spikes confirm momentum at 
extreme RSI readings. This strategy aims to capture mean reversion in ranging 
markets and trend continuation in trending markets by filtering RSI signals 
with the higher timeframe trend. Target: 7-25 trades/year per symbol (30-100 total).
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
    
    # Load 1d data for RSI calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate RSI(14) on 1d data
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])  # First average of first 14 periods
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_1d = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))
    
    # Align RSI to lower timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Load 1w data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # RSI levels
        rsi = rsi_1d_aligned[i]
        ema_trend = ema_34_1w_aligned[i]
        close_price = close[i]
        
        if position == 0:
            # Long: RSI oversold (<30) in uptrend (price > weekly EMA) with volume confirmation
            if (rsi < 30 and 
                close_price > ema_trend and 
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) in downtrend (price < weekly EMA) with volume confirmation
            elif (rsi > 70 and 
                  close_price < ema_trend and 
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: RSI returns to neutral zone (40-60) or contrary signal
            exit_signal = False
            
            if position == 1:
                # Exit long: RSI reaches 50 or bearish reversal signal
                if rsi >= 50 or (rsi > 70 and close_price < ema_trend):
                    exit_signal = True
            else:  # position == -1
                # Exit short: RSI reaches 50 or bullish reversal signal
                if rsi <= 50 or (rsi < 30 and close_price > ema_trend):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_RSI_1wEMA_Trend_Volume"
timeframe = "1d"
leverage = 1.0