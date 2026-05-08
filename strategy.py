#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h trend filter and volume spike confirmation.
# Long when RSI < 30, 4h EMA50 rising, and volume > 2x 20-period average.
# Short when RSI > 70, 4h EMA50 falling, and volume > 2x 20-period average.
# Exit when RSI crosses back to neutral (40 for long, 60 for short).
# This strategy targets overextended moves in both bull and bear markets by combining
# short-term mean reversion (RSI) with trend alignment (4h EMA50) and volume confirmation.
# Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drift.
# Works in bull markets by buying dips in uptrends and in bear markets by selling rallies in downtrends.

name = "1h_RSI_MeanReversion_4hEMA50_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h EMA50 for trend filter
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 4h EMA50 direction
    ema50_rising = np.zeros_like(ema50_4h_aligned, dtype=bool)
    ema50_falling = np.zeros_like(ema50_4h_aligned, dtype=bool)
    ema50_rising[1:] = ema50_4h_aligned[1:] > ema50_4h_aligned[:-1]
    ema50_falling[1:] = ema50_4h_aligned[1:] < ema50_4h_aligned[:-1]
    
    # Volume filter: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma20)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Sufficient warmup for EMA50 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(ema50_rising[i]) or np.isnan(ema50_falling[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(session_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: RSI < 30, 4h EMA50 rising, volume filter
            long_cond = (rsi[i] < 30) and ema50_rising[i] and volume_filter[i]
            # Short conditions: RSI > 70, 4h EMA50 falling, volume filter
            short_cond = (rsi[i] > 70) and ema50_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: RSI crosses back above 40
            if rsi[i] > 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RSI crosses back below 60
            if rsi[i] < 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals