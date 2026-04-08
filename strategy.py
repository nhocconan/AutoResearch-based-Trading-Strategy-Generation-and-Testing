#!/usr/bin/env python3
# 4h_1d_volume_confirmation_momentum_v1
# Hypothesis: Trade 4h momentum with 1d trend filter and volume confirmation.
# In bull markets, buy when price crosses above 1d EMA50 with volume surge; in bear markets, sell when price crosses below 1d EMA50 with volume surge.
# Uses RSI to avoid overbought/oversold extremes and ATR-based stops to manage risk.
# Target: 20-50 trades/year with strict entry conditions to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_volume_confirmation_momentum_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # RSI for overbought/oversold filter (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # ATR for volatility and stop (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: 4h volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 2.0 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: Price below 1d EMA50 OR stoploss hit
            if close[i] < ema50_1d_aligned[i] or close[i] < high[i] - 2.5 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above 1d EMA50 OR stoploss hit
            if close[i] > ema50_1d_aligned[i] or close[i] > low[i] + 2.5 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price crosses above 1d EMA50 with RSI < 70 and volume surge
            if (close[i] > ema50_1d_aligned[i] and 
                close[i-1] <= ema50_1d_aligned[i-1] and  # Cross above
                rsi[i] < 70 and 
                vol_surge):
                position = 1
                signals[i] = 0.25
            # Short entry: Price crosses below 1d EMA50 with RSI > 30 and volume surge
            elif (close[i] < ema50_1d_aligned[i] and 
                  close[i-1] >= ema50_1d_aligned[i-1] and  # Cross below
                  rsi[i] > 30 and 
                  vol_surge):
                position = -1
                signals[i] = -0.25
    
    return signals