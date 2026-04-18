#/usr/bin/env python3
"""
1d_KAMA_Trend_Filter
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) to capture adaptive trend direction, enter long when price crosses above KAMA with volume confirmation and bullish market regime (ADX > 20), short when price crosses below KAMA with volume confirmation and bearish regime (ADX > 20). Exit on opposite KAMA cross. Designed for low trade frequency to minimize fee drag while capturing sustained trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on daily close
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # KAMA parameters
    fast = 2
    slow = 30
    lookback = 10
    
    # Calculate efficiency ratio
    change = np.abs(np.diff(close_1d, lookback))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=1)
    er = np.zeros_like(close_1d)
    er[lookback:] = change[lookback:] / volatility[lookback:]
    er[volatility == 0] = 0
    
    # Calculate smoothing constant
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to 1d timeframe (already on 1d, but need to align to original price index)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Volume spike: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # ADX trend strength filter (using 1d data)
    # Calculate +DM, -DM, TR
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    
    # Pad arrays to match length
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    adx_period = 14
    atr = wilders_smoothing(tr, adx_period)
    plus_di = 100 * wilders_smoothing(plus_dm, adx_period) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, adx_period) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, adx_period)
    
    # Align ADX to 1d timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)  # Need KAMA, volume MA, and ADX
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(volume_spike[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama_aligned[i]
        vol_spike = volume_spike[i]
        adx_val = adx_aligned[i]
        
        # Only trade when ADX indicates trending market (ADX > 20)
        if adx_val > 20:
            if position == 0:
                # Long: price crosses above KAMA with volume spike
                if price > kama_val and close[i-1] <= kama_aligned[i-1] and vol_spike:
                    signals[i] = 0.25
                    position = 1
                # Short: price crosses below KAMA with volume spike
                elif price < kama_val and close[i-1] >= kama_aligned[i-1] and vol_spike:
                    signals[i] = -0.25
                    position = -1
            
            elif position == 1:
                signals[i] = 0.25
                # Exit: price crosses below KAMA
                if price < kama_val and close[i-1] >= kama_aligned[i-1]:
                    signals[i] = 0.0
                    position = 0
            
            elif position == -1:
                signals[i] = -0.25
                # Exit: price crosses above KAMA
                if price > kama_val and close[i-1] <= kama_aligned[i-1]:
                    signals[i] = 0.0
                    position = 0
        else:
            # In ranging market (ADX <= 20), stay flat
            signals[i] = 0.0
            position = 0
    
    return signals

name = "1d_KAMA_Trend_Filter"
timeframe = "1d"
leverage = 1.0