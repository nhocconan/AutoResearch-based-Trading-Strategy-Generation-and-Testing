#!/usr/bin/env python3

"""
Hypothesis: 4-hour RSI divergence with 1-day trend filter and volume confirmation.
RSI divergence captures momentum exhaustion at trend extremes.
The 1-day trend filter ensures trades align with the daily trend to avoid counter-trend trades.
Volume spikes confirm institutional participation at reversal points.
This strategy aims to capture reversals in both bull and bear markets by
trading RSI divergences with trend and volume confirmation.
Target: 19-50 trades/year per symbol (75-200 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = np.diff(prices, prepend=prices[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def detect_divergence(price, rsi, lookback=14):
    """Detect bullish and bearish RSI divergence"""
    n = len(price)
    bullish_div = np.zeros(n, dtype=bool)
    bearish_div = np.zeros(n, dtype=bool)
    
    for i in range(lookback, n):
        # Bullish divergence: price makes lower low, RSI makes higher low
        if price[i] < price[i-lookback:i].min() and rsi[i] > rsi[i-lookback:i].min():
            bullish_div[i] = True
        # Bearish divergence: price makes higher high, RSI makes lower high
        if price[i] > price[i-lookback:i].max() and rsi[i] < rsi[i-lookback:i].max():
            bearish_div[i] = True
            
    return bullish_div, bearish_div

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h price data for RSI calculation - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate RSI on 4h data
    rsi_4h = calculate_rsi(df_4h['close'].values, 14)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Detect RSI divergence on 4h
    bullish_div_4h, bearish_div_4h = detect_divergence(
        df_4h['close'].values, rsi_4h, 14
    )
    bullish_div_4h_aligned = align_htf_to_ltf(prices, df_4h, bullish_div_4h.astype(float))
    bearish_div_4h_aligned = align_htf_to_ltf(prices, df_4h, bearish_div_4h.astype(float))
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA for trend filter (50-period)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i]) or np.isnan(bullish_div_4h_aligned[i]) or
            np.isnan(bearish_div_4h_aligned[i])):
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
            # Long: bullish RSI divergence, above 1d EMA, volume spike
            if (bullish_div_4h_aligned[i] and                    # Bullish RSI divergence
                close[i] > ema_50_1d_aligned[i] and              # Above 1d EMA (bullish trend)
                volume[i] > 1.8 * vol_avg_20[i]):                # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: bearish RSI divergence, below 1d EMA, volume spike
            elif (bearish_div_4h_aligned[i] and                  # Bearish RSI divergence
                  close[i] < ema_50_1d_aligned[i] and            # Below 1d EMA (bearish trend)
                  volume[i] > 1.8 * vol_avg_20[i]):              # Volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit: RSI returns to neutral zone or crosses 1d EMA
            exit_signal = False
            
            if position == 1:
                # Exit long: RSI > 70 or price crosses below 1d EMA
                if rsi_4h_aligned[i] > 70 or close[i] < ema_50_1d_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: RSI < 30 or price crosses above 1d EMA
                if rsi_4h_aligned[i] < 30 or close[i] > ema_50_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_RSI_Divergence_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0