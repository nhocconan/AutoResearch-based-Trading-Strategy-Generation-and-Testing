#!/usr/bin/env python3
"""
1h RSI Pullback with 4h/1d Trend and Volume Confirmation v1
Hypothesis: In strong trends (4h/1d aligned), RSI pullbacks on 1h provide high-probability entries.
Volume confirms institutional participation. Works in bull/bear by following higher timeframe trend.
Target: 15-37 trades/year (60-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_rsi_pullback_4h1d_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for trend and RSI
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate RSI on 4h (14-period)
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_4h = calculate_rsi(close_4h, 14)
    
    # Calculate EMA on 4h and 1d for trend
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume filter on 4h (>1.3x 20-period average)
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_filter_4h = volume_4h > (vol_ma_4h * 1.3)
    
    # Align to 1h timeframe
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    vol_filter_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_filter_4h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_filter_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend alignment (both 4h and 1d must agree)
        bullish_trend = close[i] > ema_4h_aligned[i] and close[i] > ema_1d_aligned[i]
        bearish_trend = close[i] < ema_4h_aligned[i] and close[i] < ema_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: RSI > 70 (overbought) or trend breaks
            if rsi_4h_aligned[i] > 70 or not bullish_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI < 30 (oversold) or trend breaks
            if rsi_4h_aligned[i] < 30 or not bearish_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Pullback long in uptrend: RSI < 40 with volume confirmation
            if (bullish_trend and 
                rsi_4h_aligned[i] < 40 and 
                vol_filter_4h_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Pullback short in downtrend: RSI > 60 with volume confirmation
            elif (bearish_trend and 
                  rsi_4h_aligned[i] > 60 and 
                  vol_filter_4h_aligned[i]):
                position = -1
                signals[i] = -0.20
    
    return signals