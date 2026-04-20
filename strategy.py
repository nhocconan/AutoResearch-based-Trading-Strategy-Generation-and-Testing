#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h RSI(14) + 1d MACD(12,26,9) + Volume Spike Filter
# - Long when RSI < 30 (oversold) AND MACD line > signal line AND volume > 1.5x 20-period average
# - Short when RSI > 70 (overbought) AND MACD line < signal line AND volume > 1.5x 20-period average
# - RSI captures mean reversion extremes; MACD confirms momentum direction; volume filter ensures conviction
# - Designed for 4h timeframe with selective entries to avoid overtrading (target: 20-50 trades/year)
# - Works in both bull/bear: buys oversold dips in uptrend, sells overbought rallies in downtrend

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for MACD calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate MACD on 1d timeframe
    ema12 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = pd.Series(close_1d).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Align 1d MACD components to 4h timeframe
    macd_line_aligned = align_htf_to_ltf(prices, df_1d, macd_line)
    signal_line_aligned = align_htf_to_ltf(prices, df_1d, signal_line)
    
    # Calculate RSI on 4h timeframe
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate volume spike filter on 4h timeframe
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(rsi[i]) or np.isnan(macd_line_aligned[i]) or np.isnan(signal_line_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        macd_val = macd_line_aligned[i]
        signal_val = signal_line_aligned[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 0:
            # Long entry: RSI oversold + MACD bullish + volume spike
            if rsi_val < 30 and macd_val > signal_val and vol_ratio > 1.5:
                signals[i] = 0.25
                position = 1
            # Short entry: RSI overbought + MACD bearish + volume spike
            elif rsi_val > 70 and macd_val < signal_val and vol_ratio > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI overbought OR MACD bearish crossover
            if rsi_val > 70 or macd_val < signal_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI oversold OR MACD bullish crossover
            if rsi_val < 30 or macd_val > signal_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI_MACD_VolumeSpike"
timeframe = "4h"
leverage = 1.0