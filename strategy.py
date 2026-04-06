#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h trend filter and volume confirmation
# Long when RSI < 30 + 4h EMA(50) up + volume > 1.5x average
# Short when RSI > 70 + 4h EMA(50) down + volume > 1.5x average
# Exit when RSI crosses 50 or EMA crosses price
# Uses 1h timeframe targeting 100-200 total trades over 4 years (25-50/year)
# Works in ranging markets by buying dips/selling rallies with trend alignment

name = "1h_rsi_meanrev_4hma_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h EMA(50) for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    ema_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: RSI crosses 50 or EMA crosses price
        if position == 1:  # long position
            if rsi[i] >= 50 or close[i] <= ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if rsi[i] <= 50 or close[i] >= ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for mean reversion with trend and volume confirmation
            # Bullish setup: RSI oversold + EMA up + volume
            if (rsi[i] < 30 and 
                ema_4h_aligned[i] > ema_4h_aligned[i-1] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.20
                position = 1
            # Bearish setup: RSI overbought + EMA down + volume
            elif (rsi[i] > 70 and 
                  ema_4h_aligned[i] < ema_4h_aligned[i-1] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.20
                position = -1
    
    return signals