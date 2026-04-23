#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian channel breakout with 1-day RSI filter and volume confirmation.
Long when price breaks above Donchian(20) high + RSI(14) < 60 (avoid overbought) + volume > 1.5x average.
Short when price breaks below Donchian(20) low + RSI(14) > 40 (avoid oversold) + volume > 1.5x average.
Exit when price crosses the midline (average of upper/lower band).
Uses volume and RSI to filter breakouts, targeting 20-40 trades/year with strong trend capture.
Works in bull markets via breakouts and in bear via short breakdowns with volume confirmation.
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
    
    # Donchian Channel (20-period) on price
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = highest_high.values
    donchian_low = lowest_low.values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 1-day RSI (14-period) for filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to RMA)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_values = rsi_1d.values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_ma_values = vol_ma.values
    
    # Align 1-day indicators to 4h
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: breakout above Donchian high + RSI not overbought + volume spike
            if (close[i] > donchian_high[i] and 
                rsi_1d_aligned[i] < 60 and 
                volume[i] > 1.5 * vol_ma_values[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: breakdown below Donchian low + RSI not oversold + volume spike
            elif (close[i] < donchian_low[i] and 
                  rsi_1d_aligned[i] > 40 and 
                  volume[i] > 1.5 * vol_ma_values[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price crosses Donchian midline
            exit_signal = False
            if position == 1:
                if close[i] < donchian_mid[i]:
                    exit_signal = True
            else:  # position == -1
                if close[i] > donchian_mid[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_Breakout_RSI_Volume"
timeframe = "4h"
leverage = 1.0