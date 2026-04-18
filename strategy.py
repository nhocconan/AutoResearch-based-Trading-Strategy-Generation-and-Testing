#!/usr/bin/env python3
"""
1d_RSI2_Breakout_With_Volume_And_Pullback_Filter
Daily RSI(2) mean reversion strategy with volume confirmation and pullback filter.
- Long: RSI(2) < 10 + pullback to 20-day EMA + volume > 1.5x 20-day average + price above 200-day EMA
- Short: RSI(2) > 90 + pullback to 20-day EMA + volume > 1.5x 20-day average + price below 200-day EMA
- Exit: RSI(2) crosses above 50 (long) or below 50 (short)
Designed for ~10-20 trades/year per symbol (40-80 total over 4 years)
Works in trending markets by buying dips in uptrends and selling rallies in downtrends
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for indicators
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # RSI(2) calculation
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (alpha = 1/period)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[1] = gain[1]  # Initialize first value
    avg_loss[1] = loss[1]
    
    for i in range(2, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 1 + gain[i]) / 2  # 2-period smoothing
        avg_loss[i] = (avg_loss[i-1] * 1 + loss[i]) / 2
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # 20-day EMA for pullback
    ema_20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 200-day EMA for trend filter
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # 20-day volume average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align daily indicators to 1d timeframe (no alignment needed since we're on 1d)
    rsi_aligned = rsi
    ema_20_aligned = ema_20
    ema_200_aligned = ema_200
    vol_ma_20_aligned = vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # need enough for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi[i]) or np.isnan(ema_20[i]) or np.isnan(ema_200[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter
        uptrend = close[i] > ema_200[i]
        downtrend = close[i] < ema_200[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        # Pullback to 20-day EMA (within 1%)
        near_ema20 = abs(close[i] - ema_20[i]) / ema_20[i] < 0.01
        
        if position == 0:
            # Long: oversold RSI + pullback + volume + uptrend
            if rsi[i] < 10 and near_ema20 and vol_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: overbought RSI + pullback + volume + downtrend
            elif rsi[i] > 90 and near_ema20 and vol_confirm and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI crosses above 50
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI crosses below 50
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_RSI2_Breakout_With_Volume_And_Pullback_Filter"
timeframe = "1d"
leverage = 1.0