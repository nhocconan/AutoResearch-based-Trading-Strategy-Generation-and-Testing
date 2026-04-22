#!/usr/bin/env python3

"""
Hypothesis: 1-hour RSI(2) mean reversion with 4-hour trend filter and volume confirmation.
Uses RSI(2) for short-term mean reversion entries, 4-hour EMA(50) for trend direction,
and volume spikes to confirm institutional participation. Designed to work in both
bull and bear markets by only taking trades in the direction of the 4-hour trend.
Target: 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4-hour data - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4-hour EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Determine 4-hour trend: price above/below EMA(50)
    bullish_trend = close_4h > ema_50_4h
    bearish_trend = close_4h < ema_50_4h
    
    # Align 4-hour EMA and trend to 1-hour timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    bullish_aligned = align_htf_to_ltf(prices, df_4h, bullish_trend.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_4h, bearish_trend.astype(float))
    
    # Calculate 1-hour RSI(2)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1-hour volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(bullish_aligned[i]) or 
            np.isnan(bearish_aligned[i]) or np.isnan(rsi[i]) or 
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
        
        if position == 0:
            # Long: RSI(2) < 10 (oversold), bullish 4h trend, volume spike
            if (rsi[i] < 10 and 
                bullish_aligned[i] > 0.5 and 
                volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = 0.20
                position = 1
            # Short: RSI(2) > 90 (overbought), bearish 4h trend, volume spike
            elif (rsi[i] > 90 and 
                  bearish_aligned[i] > 0.5 and 
                  volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: RSI returns to neutral (40-60) or opposite extreme
            exit_signal = False
            
            if position == 1:
                # Exit long: RSI > 40 (recovered from oversold)
                if rsi[i] > 40:
                    exit_signal = True
            else:  # position == -1
                # Exit short: RSI < 60 (declined from overbought)
                if rsi[i] < 60:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_RSI2_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0