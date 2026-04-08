#!/usr/bin/env python3
"""
4h KAMA trend with 1d RSI filter and volume confirmation
Hypothesis: KAMA adapts to market noise, reducing false signals in choppy markets.
Combined with 1d RSI for trend strength and volume confirmation, this filters trades
to high-probability moments while maintaining low trade frequency for 4h timeframe.
Works in bull markets via KAMA uptrend + RSI > 50, and in bear markets via
KAMA downtrend + RSI < 50, avoiding whipsaw during sideways periods.
"""

name = "4h_kama_1d_rsi_volume_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for RSI (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 14-period RSI for 1d data
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # First average gain/loss
    avg_gain = np.concatenate([[np.nan], np.full(13, np.nan), [np.mean(gain[:14])]])
    avg_loss = np.concatenate([[np.nan], np.full(13, np.nan), [np.mean(loss[:14])]])
    
    # Wilder's smoothing
    for i in range(15, len(close_1d)):
        avg_gain = np.append(avg_gain, (avg_gain[-1] * 13 + gain[i-1]) / 14)
        avg_loss = np.append(avg_loss, (avg_loss[-1] * 13 + loss[i-1]) / 14)
    
    # Handle initial NaN values
    avg_gain[:14] = np.nan
    avg_loss[:14] = np.nan
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) for 4h data
    # Efficiency Ratio
    change = np.abs(np.diff(close, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    
    # Handle array alignment
    change = np.concatenate([[np.nan]*10, change])
    volatility = np.concatenate([[np.nan]*10, volatility])
    
    er = np.divide(change, volatility, out=np.full_like(change, np.nan), where=volatility!=0)
    
    # Smoothing constants
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start with first value
    
    for i in range(10, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # 20-period volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 40  # Need KAMA and RSI buffers
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi_1d[i]) or 
            np.isnan(vol_avg_20[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned 1d value for current 4h bar
        rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)[i]
        
        # Trend filter: KAMA direction
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        # RSI filter: 1d RSI for trend strength
        rsi_bullish = rsi_1d_aligned > 50
        rsi_bearish = rsi_1d_aligned < 50
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirm = volume[i] > 1.3 * vol_avg_20[i]
        
        if position == 1:  # Long position
            # Exit: KAMA turns down
            if kama_falling:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: KAMA turns up
            if kama_rising:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade with volume confirmation
            if volume_confirm:
                # Long entry: KAMA rising AND 1d RSI bullish
                if kama_rising and rsi_bullish:
                    position = 1
                    signals[i] = 0.25
                # Short entry: KAMA falling AND 1d RSI bearish
                elif kama_falling and rsi_bearish:
                    position = -1
                    signals[i] = -0.25
    
    return signals