#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily KAMA trend with daily RSI mean reversion and volume confirmation.
# Long when price crosses above KAMA AND RSI < 30 (oversold) with volume spike.
# Short when price crosses below KAMA AND RSI > 70 (overbought) with volume spike.
# Exit when price returns to KAMA or RSI reaches opposite extreme (70 for long, 30 for short).
# Uses daily KAMA for adaptive trend, daily RSI for mean reversion, volume for confirmation.
# Target: 25-35 trades/year per symbol (100-140 total over 4 years) to minimize fee drift.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily KAMA (adaptive moving average)
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # This needs correction
    
    # Correct calculation for ER and KAMA
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        volatility[i] = np.sum(np.abs(np.diff(close_1d[max(0, i-9):i+1])))
    
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # for EMA(2)
    slow_sc = 2 / (30 + 1)  # for EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate daily RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align indicators to lower timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20)  # Need KAMA/RSI and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for mean reversion entries
            # Long: price crosses above KAMA AND RSI < 30 (oversold)
            if (close[i] > kama_aligned[i] and 
                close[i-1] <= kama_aligned[i-1] and  # crossed above
                rsi_aligned[i] < 30 and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price crosses below KAMA AND RSI > 70 (overbought)
            elif (close[i] < kama_aligned[i] and 
                  close[i-1] >= kama_aligned[i-1] and  # crossed below
                  rsi_aligned[i] > 70 and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to KAMA or RSI reaches overbought
            if (close[i] <= kama_aligned[i] or 
                rsi_aligned[i] >= 70):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to KAMA or RSI reaches oversold
            if (close[i] >= kama_aligned[i] or 
                rsi_aligned[i] <= 30):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_DailyKAMA_RSI_MeanReversion_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0