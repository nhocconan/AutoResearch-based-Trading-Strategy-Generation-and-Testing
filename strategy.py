#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h 13-period RSI with 1-day high-low range breakout and volume confirmation
# Enters long when RSI < 30 and price breaks above 1-day high with volume spike
# Enters short when RSI > 70 and price breaks below 1-day low with volume spike
# Uses 1-day ATR for stop loss and position sizing
# Target: 15-35 trades/year to avoid fee drag, works in both bull and bear via mean reversion
name = "12h_RSI_RangeBreakout_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for multi-timeframe analysis (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1-day high and low for range breakout
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    
    # 1-day ATR for volatility filter and stop
    high_1d_arr = df_1d['high'].values
    low_1d_arr = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    tr_1d = np.maximum(high_1d_arr - low_1d_arr, 
                       np.maximum(np.abs(high_1d_arr - np.roll(close_1d_arr, 1)),
                                  np.abs(low_1d_arr - np.roll(close_1d_arr, 1))))
    tr_1d[0] = high_1d_arr[0] - low_1d_arr[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 12-period RSI for mean reversion signal
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume filter: current volume > 2.0 x 20-period average
    if len(volume) >= 20:
        avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    else:
        avg_volume = np.full_like(volume, volume[0])
    volume_filter = volume > 2.0 * avg_volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(high_1d_aligned[i]) or np.isnan(low_1d_aligned[i]) or \
           np.isnan(atr_1d_aligned[i]) or np.isnan(rsi_values[i]) or \
           np.isnan(avg_volume[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_1d_aligned[i]
        
        if position == 0:
            # Long: RSI oversold (<30) + price breaks above 1-day high + volume spike
            if rsi_values[i] < 30 and price > high_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) + price breaks below 1-day low + volume spike
            elif rsi_values[i] > 70 and price < low_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: RSI returns to neutral (40-60) or price drops below 1-day low or ATR stop
            if (rsi_values[i] >= 40 and rsi_values[i] <= 60) or \
               price < low_1d_aligned[i] or \
               price < high[i-1] - 1.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: RSI returns to neutral (40-60) or price rises above 1-day high or ATR stop
            if (rsi_values[i] >= 40 and rsi_values[i] <= 60) or \
               price > high_1d_aligned[i] or \
               price < low[i-1] + 1.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals