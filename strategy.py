#!/usr/bin/env python3
"""
12h_RSI_Reversal_VolumeFilter
12h strategy using RSI mean reversion with volume confirmation and daily trend filter.
- Long: RSI < 30 + volume > 1.5x daily avg + daily EMA50 > EMA200
- Short: RSI > 70 + volume > 1.5x daily avg + daily EMA50 < EMA200
- Exit: RSI crosses back above 50 (long) or below 50 (short)
Designed for ~20-30 trades/year per symbol (80-120 total over 4 years)
Works in bull markets (dips in uptrend) and bear markets (bounces in downtrend)
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
    
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
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
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for EMA200 and RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = ema_50_aligned[i] > ema_200_aligned[i]
        downtrend = ema_50_aligned[i] < ema_200_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # RSI conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_exit_long = rsi[i] > 50
        rsi_exit_short = rsi[i] < 50
        
        if position == 0:
            # Long: oversold + volume + uptrend
            if rsi_oversold and vol_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: overbought + volume + downtrend
            elif rsi_overbought and vol_confirm and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI crosses above 50 or trend change
            if rsi_exit_long or not uptrend:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI crosses below 50 or trend change
            if rsi_exit_short or not downtrend:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_RSI_Reversal_VolumeFilter"
timeframe = "12h"
leverage = 1.0