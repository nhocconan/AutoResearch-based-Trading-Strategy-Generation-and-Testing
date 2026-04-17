#!/usr/bin/env python3
"""
4h_RSI_Stoch_Divergence_v1
RSI(14) divergence + Stochastic(14,3,3) oversold/overbought with volume confirmation.
Long: Bullish RSI divergence + Stoch < 20 + volume > 1.5x average.
Short: Bearish RSI divergence + Stoch > 80 + volume > 1.5x average.
Exit when RSI crosses 50 or Stoch reverses.
Uses 1d EMA200 for trend filter: only long when price > EMA200, short when price < EMA200.
Designed to catch reversals in both trending and ranging markets.
Target: 50-120 total trades over 4 years (12-30/year).
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
    
    # === RSI(14) ===
    delta = np.diff(close)
    delta = np.concatenate([[0], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === Stochastic(14,3,3) ===
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    d_percent = pd.Series(k_percent).rolling(window=3, min_periods=3).mean().values
    
    # === Volume average (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1d EMA200 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or 
            np.isnan(k_percent[i]) or 
            np.isnan(d_percent[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(ema_200_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Bullish RSI divergence: price making lower low, RSI making higher low
            bull_div = False
            if i >= 3:
                # Check last 3 bars for divergence
                if (close[i] < close[i-1] < close[i-2] and 
                    rsi[i] > rsi[i-1] > rsi[i-2]):
                    bull_div = True
            
            # Bearish RSI divergence: price making higher high, RSI making lower high
            bear_div = False
            if i >= 3:
                if (close[i] > close[i-1] > close[i-2] and 
                    rsi[i] < rsi[i-1] < rsi[i-2]):
                    bear_div = True
            
            # Long: Bullish divergence + Stoch oversold + volume + price above EMA200
            if (bull_div and 
                k_percent[i] < 20 and 
                vol_confirmed and 
                close[i] > ema_200_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: Bearish divergence + Stoch overbought + volume + price below EMA200
            elif (bear_div and 
                  k_percent[i] > 80 and 
                  vol_confirmed and 
                  close[i] < ema_200_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: RSI crosses below 50 OR Stoch crosses above 50
            if (rsi[i] < 50 or 
                k_percent[i] > 50):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI crosses above 50 OR Stoch crosses below 50
            if (rsi[i] > 50 or 
                k_percent[i] < 50):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI_Stoch_Divergence_v1"
timeframe = "4h"
leverage = 1.0