#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4-period RSI with 4h trend filter and volume confirmation
# Long when RSI < 30 (oversold) AND price > 4h EMA(50) AND volume > 1.5x 20-period average
# Short when RSI > 70 (overbought) AND price < 4h EMA(50) AND volume > 1.5x 20-period average
# Exit when RSI crosses back to neutral (40 for long exit, 60 for short exit)
# Uses 1h timeframe for entry timing, 4h for trend direction to reduce false signals
# Target: 60-150 total trades over 4 years (15-37/year) for optimal 1h performance

name = "1h_rsi4_4h_ema_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4-period RSI
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/4, min_periods=4, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/4, min_periods=4, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # 4h EMA(50) trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: RSI crosses back to neutral
        if position == 1:  # long position
            if rsi[i] > 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if rsi[i] < 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with trend filter and volume confirmation
            # Long: RSI < 30 (oversold) AND price > 4h EMA AND volume confirmation
            if (rsi[i] < 30 and close[i] > ema_4h_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.20
                position = 1
            # Short: RSI > 70 (overbought) AND price < 4h EMA AND volume confirmation
            elif (rsi[i] > 70 and close[i] < ema_4h_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.20
                position = -1
    
    return signals