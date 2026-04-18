#!/usr/bin/env python3
"""
12h_Volume_Weighted_RSI_MeanReversion_v1
Hypothesis: In 12h timeframe, RSI mean reversion with volume confirmation works across market regimes.
Long when RSI < 30 and volume > 1.5x average, short when RSI > 70 and volume > 1.5x average.
Uses 1-week trend filter to avoid counter-trend trades in strong trends.
Target: 20-30 trades/year by requiring RSI extremes + volume confirmation + trend alignment.
Works in both bull and bear markets by fading extremes only when aligned with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate RSI(14)
    def calculate_rsi(close_prices, period=14):
        delta = np.diff(close_prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close_prices, np.nan)
        avg_loss = np.full_like(close_prices, np.nan)
        
        if len(close_prices) >= period + 1:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
            
            for i in range(period + 1, len(close_prices)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA(34) for trend filter
    if len(close_1w) >= 34:
        ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False).mean().values
    else:
        ema_1w = np.full_like(close_1w, np.nan)
    
    # Align weekly EMA to 12h timeframe
    ema_1w_12h = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 20, 34) + 1  # Ensure we have enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_1w_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above weekly EMA for longs, below for shorts
        bullish_trend = close[i] > ema_1w_12h[i]
        bearish_trend = close[i] < ema_1w_12h[i]
        
        if position == 0:
            # Long: RSI oversold with volume confirmation and bullish trend
            if rsi[i] < 30 and vol_confirm and bullish_trend:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought with volume confirmation and bearish trend
            elif rsi[i] > 70 and vol_confirm and bearish_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral (50) or trend changes
            if rsi[i] >= 50 or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral (50) or trend changes
            if rsi[i] <= 50 or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Volume_Weighted_RSI_MeanReversion_v1"
timeframe = "12h"
leverage = 1.0