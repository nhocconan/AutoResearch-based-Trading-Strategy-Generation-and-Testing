#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend filter + 1d RSI mean reversion + volume confirmation
# Long when KAMA indicates uptrend + RSI < 30 (oversold) + volume > 1.5x avg
# Short when KAMA indicates downtrend + RSI > 70 (overbought) + volume > 1.5x avg
# Exit when RSI crosses 50 in opposite direction
# KAMA adapts to market noise, reducing whipsaw in ranging markets
# RSI provides mean-reversion entries in both bull and bear markets
# Volume confirmation ensures institutional participation
# Target: 50-150 trades over 4 years with controlled frequency

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h KAMA (ER=10, smoothing=2,30)
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close, n=10, prepend=close[:10]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d RSI (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(30, n):
        # Get aligned 1d RSI
        rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)[i]
        
        # Check for NaN values
        if (np.isnan(kama[i]) or np.isnan(vol_ma[i]) or np.isnan(rsi_aligned)):
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:  # No position - look for entries
            if volume_confirm:
                # Long: KAMA uptrend (price > KAMA) + RSI oversold
                if close[i] > kama[i] and rsi_aligned < 30:
                    position = 1
                    signals[i] = position_size
                # Short: KAMA downtrend (price < KAMA) + RSI overbought
                elif close[i] < kama[i] and rsi_aligned > 70:
                    position = -1
                    signals[i] = -position_size
        elif position == 1:  # Long position - exit when RSI crosses above 50
            if rsi_aligned > 50:
                position = 0
                signals[i] = 0.0
        elif position == -1:  # Short position - exit when RSI crosses below 50
            if rsi_aligned < 50:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_KAMA_RSI_MeanRev_Volume"
timeframe = "12h"
leverage = 1.0