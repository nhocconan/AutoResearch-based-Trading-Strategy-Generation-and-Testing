#!/usr/bin/env python3
"""
12h_kama_trend_1w_rsi_volume_v2
Hypothesis: On 12-hour timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
with weekly RSI filter to avoid overextended moves and volume confirmation for institutional participation.
Long when KAMA turns up, weekly RSI < 50 (not overbought), and volume > 1.5x 20-period average.
Short when KAMA turns down, weekly RSI > 50 (not oversold), and volume > 1.5x 20-period average.
Exit when KAMA reverses direction. Designed for 15-30 trades/year to minimize fee drag while capturing trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_kama_trend_1w_rsi_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA on 12h close
    def kama(close, period=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=period))
        volatility = np.abs(np.diff(close)).cumsum()
        volatility = volatility[period-1:] - np.concatenate(([0], volatility[:-period+1]))
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.full_like(close, np.nan)
        kama[period-1] = close[period-1]
        for i in range(period, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_values = kama(close, period=10, fast=2, slow=30)
    
    # Get weekly data for RSI filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate weekly RSI
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1w = 100 - (100 / (1 + rs))
    # Prepend NaN for first element
    rsi_1w = np.concatenate(([np.nan], rsi_1w))
    
    # Align weekly RSI to 12h timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Volume filter: 20-period average on 12h timeframe
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(30, 14), n):
        # Skip if data not available
        if (np.isnan(kama_values[i]) or np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: KAMA turns down
            if kama_values[i] < kama_values[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: KAMA turns up
            if kama_values[i] > kama_values[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation
            if vol_ok:
                # Long: KAMA turning up, weekly RSI < 50 (not overbought)
                if kama_values[i] > kama_values[i-1] and rsi_1w_aligned[i] < 50:
                    position = 1
                    signals[i] = 0.25
                # Short: KAMA turning down, weekly RSI > 50 (not oversold)
                elif kama_values[i] < kama_values[i-1] and rsi_1w_aligned[i] > 50:
                    position = -1
                    signals[i] = -0.25
    
    return signals