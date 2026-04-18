#!/usr/bin/env python3
"""
1h RSI Divergence + Volume Spike + 4h Trend Filter
Hypothesis: RSI divergence at key levels with volume confirmation signals exhaustion moves.
4h EMA50 acts as trend filter to avoid counter-trend trades. Designed for 15-35 trades/year.
Works in bull markets (buy oversold bounces) and bear markets (sell overbought bounces).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI calculation
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
        rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Find RSI peaks and troughs for divergence
    def find_divergences(price, rsi, lookback=5):
        bullish_div = np.zeros(len(price), dtype=bool)
        bearish_div = np.zeros(len(price), dtype=bool)
        
        for i in range(lookback, len(price) - lookback):
            # Bullish divergence: price makes lower low, RSI makes higher low
            if (price[i] == np.min(price[i-lookback:i+lookback+1]) and 
                rsi[i] == np.max(rsi[i-lookback:i+lookback+1])):
                # Look for prior swing low
                for j in range(i-lookback, max(0, i-2*lookback), -1):
                    if (price[j] == np.min(price[j-lookback:j+lookback+1]) and 
                        price[j] > price[i] and rsi[j] < rsi[i]):
                        bullish_div[i] = True
                        break
            
            # Bearish divergence: price makes higher high, RSI makes lower high
            if (price[i] == np.max(price[i-lookback:i+lookback+1]) and 
                rsi[i] == np.min(rsi[i-lookback:i+lookback+1])):
                # Look for prior swing high
                for j in range(i-lookback, max(0, i-2*lookback), -1):
                    if (price[j] == np.max(price[j-lookback:j+lookback+1]) and 
                        price[j] < price[i] and rsi[j] > rsi[i]):
                        bearish_div[i] = True
                        break
        
        return bullish_div, bearish_div
    
    bullish_div, bearish_div = find_divergences(close, rsi, 5)
    
    # Volume spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Get 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Session filter: 8-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_50 = ema_50_4h_aligned[i]
        
        if position == 0:
            # Long: bullish RSI divergence + volume spike + above 4h EMA50
            if bullish_div[i] and volume_spike[i] and price > ema_50:
                signals[i] = 0.20
                position = 1
            # Short: bearish RSI divergence + volume spike + below 4h EMA50
            elif bearish_div[i] and volume_spike[i] and price < ema_50:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            signals[i] = 0.20
            # Exit: RSI > 70 (overbought) or price below 4h EMA50
            if rsi[i] > 70 or price < ema_50:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.20
            # Exit: RSI < 30 (oversold) or price above 4h EMA50
            if rsi[i] < 30 or price > ema_50:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_RSIDivergence_VolumeSpike_4hEMA50"
timeframe = "1h"
leverage = 1.0