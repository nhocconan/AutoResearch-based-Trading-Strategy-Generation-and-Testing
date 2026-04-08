#!/usr/bin/env python3
# 4h_rsi_mean_reversion_volume_filter_v1
# Hypothesis: RSI mean reversion with volume confirmation on 4h timeframe. 
# Long when RSI crosses above 30 with volume > 1.5x average. 
# Short when RSI crosses below 70 with volume > 1.5x average.
# Exit when RSI crosses 50 (mean reversion complete).
# Uses 4h RSI and volume filters to avoid overtrading.
# Target: 20-50 trades/year with strict entry conditions.

import numpy as np
import pandas as pd

name = "4h_rsi_mean_reversion_volume_filter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI(14) calculation
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use exponential moving average for RSI
    gain_ema = np.full(n, np.nan)
    loss_ema = np.full(n, np.nan)
    gain_ema[rsi_period-1] = np.mean(gain[:rsi_period])
    loss_ema[rsi_period-1] = np.mean(loss[:rsi_period])
    
    for i in range(rsi_period, n):
        gain_ema[i] = (gain[i] + (gain_ema[i-1] * (rsi_period - 1))) / rsi_period
        loss_ema[i] = (loss[i] + (loss_ema[i-1] * (rsi_period - 1))) / rsi_period
    
    rs = np.divide(gain_ema, loss_ema, out=np.full(n, np.nan), where=loss_ema!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(rsi_period, vol_ma_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses above 50 (mean reversion complete)
            if rsi[i] >= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI crosses below 50 (mean reversion complete)
            if rsi[i] <= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: RSI crosses above 30 with volume surge
            if rsi[i] > 30 and rsi[i-1] <= 30 and vol_surge[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: RSI crosses below 70 with volume surge
            elif rsi[i] < 70 and rsi[i-1] >= 70 and vol_surge[i]:
                position = -1
                signals[i] = -0.25
    
    return signals