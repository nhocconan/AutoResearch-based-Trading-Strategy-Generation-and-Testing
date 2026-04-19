#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h EMA crossover with 12h RSI filter and volume confirmation.
# Long when: 6h EMA(9) crosses above EMA(21), 12h RSI < 70, volume > 1.5x 20-period average
# Short when: 6h EMA(9) crosses below EMA(21), 12h RSI > 30, volume > 1.5x 20-period average
# Exit when opposite EMA crossover occurs.
# Designed for ~20-30 trades/year per symbol. Works in both bull and bear markets by using RSI to avoid overextended entries.
name = "6h_EMA_Cross_RSI12_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12-hour data for RSI filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate RSI on 12h data
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(prices, np.nan)
        avg_loss = np.full_like(prices, np.nan)
        
        # First average
        if len(gain) >= period:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
        
        # Wilder's smoothing
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_12h = calculate_rsi(close_12h, 14)
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # 6h EMA(9) and EMA(21)
    ema9 = pd.Series(close).ewm(span=9, adjust=False).values
    ema21 = pd.Series(close).ewm(span=21, adjust=False).values
    
    # Volume average (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_12h_aligned[i]) or np.isnan(ema9[i]) or 
            np.isnan(ema21[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi_12h_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # EMA crossover signals
        ema9_prev = ema9[i-1] if i > 0 else ema9[i]
        ema21_prev = ema21[i-1] if i > 0 else ema21[i]
        ema9_cross_above = ema9_prev <= ema21_prev and ema9[i] > ema21[i]
        ema9_cross_below = ema9_prev >= ema21_prev and ema9[i] < ema21[i]
        
        if position == 0:
            # Long entry: EMA9 crosses above EMA21, RSI not overbought, volume confirmation
            if ema9_cross_above and rsi_val < 70 and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short entry: EMA9 crosses below EMA21, RSI not oversold, volume confirmation
            elif ema9_cross_below and rsi_val > 30 and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: EMA9 crosses below EMA21
            if ema9_cross_below:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: EMA9 crosses above EMA21
            if ema9_cross_above:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals