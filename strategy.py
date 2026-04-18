#!/usr/bin/env python3
"""
12h_Combined_Signal_With_Volume_Confirmation
Combines multiple entry signals (breakout, pullback, momentum) with volume confirmation and 1d trend filter.
Designed for 12-30 trades per year per symbol (48-120 total over 4 years).
Works in bull markets (breakout continuation) and bear markets (pullback/reversal entries).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for breakout levels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, high_1w)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, low_1w)
    
    # Get daily data for trend filter and volume average
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily EMA50 and EMA200 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Daily volume average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Daily RSI(14) for pullback signals
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14.fillna(50).values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for EMA200 and RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or np.isnan(rsi_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = ema_50_aligned[i] > ema_200_aligned[i]
        downtrend = ema_50_aligned[i] < ema_200_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Entry signals
        breakout_up = close[i] > weekly_high_aligned[i]
        breakdown_down = close[i] < weekly_low_aligned[i]
        pullback_long = rsi_14_aligned[i] < 30 and close[i] > low[i]  # RSI oversold + price above low
        pullback_short = rsi_14_aligned[i] > 70 and close[i] < high[i]  # RSI overbought + price below high
        
        if position == 0:
            # Long: uptrend + volume + (breakout OR pullback)
            if uptrend and vol_confirm and (breakout_up or pullback_long):
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + (breakdown OR pullback)
            elif downtrend and vol_confirm and (breakdown_down or pullback_short):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change, RSI overbought, or breakdown
            if not uptrend or rsi_14_aligned[i] > 70 or breakdown_down:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change, RSI oversold, or breakout
            if not downtrend or rsi_14_aligned[i] < 30 or breakout_up:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Combined_Signal_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0