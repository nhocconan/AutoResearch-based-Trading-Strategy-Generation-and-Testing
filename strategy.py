#!/usr/bin/env python3
"""
Hypothesis: 1h RSI(14) mean reversion with 4h trend filter and volume confirmation.
Long when RSI < 30 (oversold) + 4h close > 4h EMA20 + volume > 1.5x average.
Short when RSI > 70 (overbought) + 4h close < 4h EMA20 + volume > 1.5x average.
Exit when RSI crosses 50 (mean reversion complete).
Designed for low trade frequency (~15-35/year) to minimize fee drag in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data for trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA20 for trend filter
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Calculate RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema20_4h_aligned[i]) or 
            np.isnan(avg_volume[i]) or volume[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        rsi_val = rsi[i]
        close_4h_val = close_4h[-1] if len(close_4h) > 0 else np.nan
        ema20_4h_val = ema20_4h_aligned[i]
        
        if np.isnan(close_4h_val) or np.isnan(ema20_4h_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        trend_up = close_4h_val > ema20_4h_val
        trend_down = close_4h_val < ema20_4h_val
        
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        if position == 0:
            # Long: RSI < 30 (oversold) + 4h uptrend + volume confirmation
            if (rsi_val < 30 and trend_up and volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: RSI > 70 (overbought) + 4h downtrend + volume confirmation
            elif (rsi_val > 70 and trend_down and volume_confirm):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: RSI crosses above 50 (mean reversion complete)
                if rsi[i] >= 50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: RSI crosses below 50 (mean reversion complete)
                if rsi[i] <= 50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_RSI14_4hEMA20_VolumeFilter"
timeframe = "1h"
leverage = 1.0