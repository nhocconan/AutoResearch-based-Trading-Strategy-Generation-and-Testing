#!/usr/bin/env python3
# 1h_mean_reversion_v1
# Hypothesis: Mean reversion on 1h with 4h trend filter and volume confirmation.
# Long when: RSI < 30, price > 4h VWAP (uptrend), volume > 1.5x average.
# Short when: RSI > 70, price < 4h VWAP (downtrend), volume > 1.5x average.
# Exit when RSI crosses back to neutral (40-60).
# Uses 4h for trend direction, 1h for entry timing. Target: 15-35 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_mean_reversion_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h RSI for mean reversion signals
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
    avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
    
    for i in range(rsi_period+1, n):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi[:rsi_period] = np.nan
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Get 4h VWAP for trend filter
    df_4h = get_htf_data(prices, '4h')
    typical_price_4h = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3
    vwap_4h = (typical_price_4h * df_4h['volume']).cumsum() / df_4h['volume'].cumsum()
    vwap_4h_array = vwap_4h.values
    vwap_4h_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h_array)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(rsi_period, vol_ma_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or np.isnan(vwap_4h_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses above 40 (mean reversion complete)
            if rsi[i] >= 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI crosses below 60 (mean reversion complete)
            if rsi[i] <= 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long entry: RSI < 30 (oversold), price > 4h VWAP (uptrend), volume surge
            if (rsi[i] < 30 and 
                close[i] > vwap_4h_aligned[i] and 
                vol_surge[i]):
                position = 1
                signals[i] = 0.20
            # Short entry: RSI > 70 (overbought), price < 4h VWAP (downtrend), volume surge
            elif (rsi[i] > 70 and 
                  close[i] < vwap_4h_aligned[i] and 
                  vol_surge[i]):
                position = -1
                signals[i] = -0.20
    
    return signals