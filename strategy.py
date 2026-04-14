#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for ATR and price position
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 50-period weekly ATR
    tr = np.zeros(len(df_1w))
    tr[0] = high_1w[0] - low_1w[0]
    for i in range(1, len(df_1w)):
        tr[i] = max(
            high_1w[i] - low_1w[i],
            abs(high_1w[i] - close_1w[i-1]),
            abs(low_1w[i] - close_1w[i-1])
        )
    
    atr_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 50:
        atr_1w[49] = np.mean(tr[:50])
        for i in range(50, len(df_1w)):
            atr_1w[i] = (atr_1w[i-1] * 49 + tr[i]) / 50
    
    atr_12h = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate weekly high-low range for normalization
    range_1w = high_1w - low_1w
    range_12h = align_htf_to_ltf(prices, df_1w, range_1w)
    
    # Calculate 12h price position within weekly range (0 to 1)
    price_pos = np.zeros(n)
    for i in range(n):
        if range_12h[i] > 0:
            price_pos[i] = (close[i] - low[i]) / range_12h[i]
        else:
            price_pos[i] = 0.5
    
    # Calculate 12-period 12h RSI for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    if len(gain) >= 12:
        avg_gain[11] = np.mean(gain[:12])
        avg_loss[11] = np.mean(loss[:12])
        for i in range(12, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 11 + gain[i]) / 12
            avg_loss[i] = (avg_loss[i-1] * 11 + loss[i]) / 12
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike detection (20-period average on 12h)
    vol_ma_20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_12h[i]) or 
            np.isnan(range_12h[i]) or
            np.isnan(rsi[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.5% of price)
        if atr_12h[i] < 0.005 * close[i]:
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 12h volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        # Volume threshold: require significant spike
        vol_threshold = 2.5
        
        if position == 0:
            # Long: Price in lower 30% of weekly range with RSI oversold and volume spike
            if (price_pos[i] < 0.3 and rsi[i] < 30 and volume_ratio > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short: Price in upper 70% of weekly range with RSI overbought and volume spike
            elif (price_pos[i] > 0.7 and rsi[i] > 70 and volume_ratio > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price moves to middle 40% of weekly range or RSI overbought
            if (price_pos[i] > 0.6 or rsi[i] > 70):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price moves to middle 40% of weekly range or RSI oversold
            if (price_pos[i] < 0.4 or rsi[i] < 30):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1w_RangePosition_RSI_Volume"
timeframe = "12h"
leverage = 1.0