#!/usr/bin/env python3
"""
4h_Close_EMA200_RSI14_Pullback
4h strategy using EMA200 trend filter + RSI(14) pullback to EMA200 with volume confirmation.
- Long: Close above EMA200 + RSI(14) < 40 + volume > 1.2x average
- Short: Close below EMA200 + RSI(14) > 60 + volume > 1.2x average
- Exit: Opposite signal or RSI crossing 50
Designed for ~20-30 trades/year per symbol (80-120 total over 4 years)
Works in bull markets (buy pullbacks in uptrend) and bear markets (sell rallies in downtrend)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA200 and volume average
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily EMA200
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate daily volume average (20-period)
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align daily data to 4h timeframe
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate RSI(14) on 4h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # need enough for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_200_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = close[i] > ema_200_aligned[i]
        downtrend = close[i] < ema_200_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.2 * vol_ma_20_aligned[i]
        
        # RSI conditions
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        rsi_neutral_cross = (rsi[i] >= 50 and rsi[i-1] < 50) or (rsi[i] <= 50 and rsi[i-1] > 50)
        
        if position == 0:
            # Long: uptrend + volume + RSI oversold
            if uptrend and vol_confirm and rsi_oversold:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + RSI overbought
            elif downtrend and vol_confirm and rsi_overbought:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change or RSI crosses above 50
            if not uptrend or rsi_neutral_cross:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change or RSI crosses below 50
            if not downtrend or rsi_neutral_cross:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Close_EMA200_RSI14_Pullback"
timeframe = "4h"
leverage = 1.0