#!/usr/bin/env python3
"""
4h_RSI_Trend_Divergence_With_Volume
Hypothesis: RSI divergence on 4h with trend confirmation (EMA50) and volume spike
Works in bull/bear: divergence signals reversals, volume confirms institutional interest
Target: 20-40 trades/year by requiring multiple confluence factors
"""
name = "4h_RSI_Trend_Divergence_With_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # RSI(14) - standard calculation
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices, prepend=prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing
        avg_gain = np.zeros_like(gain)
        avg_loss = np.zeros_like(loss)
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period+1, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # EMA(50) for trend filter
    def calculate_ema(data, period):
        ema = np.zeros_like(data)
        alpha = 2.0 / (period + 1)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema
    
    # 4h data for indicators
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    rsi_4h = calculate_rsi(df_4h['close'].values, 14)
    ema50_4h = calculate_ema(df_4h['close'].values, 50)
    
    # Align to lower timeframe (though we're on 4h, this ensures proper handling)
    rsi_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    ema50_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume confirmation: volume > 2.0 * 20-period average (strict for fewer trades)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_aligned[i]) or np.isnan(ema50_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        bullish_div = False
        if i >= 5:  # Look back for divergence
            # Find recent low in price
            price_low_idx = i - np.argmin(low[i-4:i+1]) - 4
            if price_low_idx >= start_idx:
                # Find another low earlier
                price_low_idx2 = price_low_idx - np.argmin(low[price_low_idx-4:price_low_idx+1]) - 4
                if price_low_idx2 >= start_idx:
                    # Check if second low is lower than first (lower low in price)
                    if low[price_low_idx2] < low[price_low_idx]:
                        # Check if RSI at second low is higher than at first (higher low in RSI)
                        if rsi_aligned[price_low_idx2] > rsi_aligned[price_low_idx]:
                            bullish_div = True
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        bearish_div = False
        if i >= 5:
            # Find recent high in price
            price_high_idx = i - np.argmax(high[i-4:i+1]) - 4
            if price_high_idx >= start_idx:
                # Find another high earlier
                price_high_idx2 = price_high_idx - np.argmax(high[price_high_idx-4:price_high_idx+1]) - 4
                if price_high_idx2 >= start_idx:
                    # Check if second high is higher than first (higher high in price)
                    if high[price_high_idx2] > high[price_high_idx]:
                        # Check if RSI at second high is lower than at first (lower high in RSI)
                        if rsi_aligned[price_high_idx2] < rsi_aligned[price_high_idx]:
                            bearish_div = True
        
        if position == 0:
            # Long: bullish divergence + price above EMA50 + volume spike
            if (bullish_div and 
                close[i] > ema50_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish divergence + price below EMA50 + volume spike
            elif (bearish_div and 
                  close[i] < ema50_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit on bearish divergence or price below EMA50
            if bearish_div or close[i] < ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit on bullish divergence or price above EMA50
            if bullish_div or close[i] > ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals