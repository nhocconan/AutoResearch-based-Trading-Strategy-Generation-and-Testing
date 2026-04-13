#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h price action above/below 1-day KAMA with volume confirmation.
# Kaufman Adaptive Moving Average (KAMA) adapts to market noise - in trending markets
# it follows price closely, in ranging markets it stays flat. This creates a dynamic
# support/resistance that respects both trending and ranging conditions.
# Combined with volume spikes to confirm breakouts and avoid false signals.
# Target: 15-30 trades per year (60-120 total over 4 years) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for KAMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # KAMA ( Kaufman Adaptive Moving Average ) - 30 period
    # ER = Efficiency Ratio, SC = Smoothing Constant
    def calculate_kama(data, period=30, fast=2, slow=30):
        kama = np.full(len(data), np.nan)
        if len(data) < period:
            return kama
            
        # Calculate change and volatility
        change = np.abs(np.diff(data, period))
        volatility = np.sum(np.abs(np.diff(data)), axis=1)
        
        # Avoid division by zero
        er = np.where(volatility != 0, change / volatility, 0)
        
        # Smoothing constant
        sc = np.power(er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1), 2)
        
        # Initialize KAMA
        kama[period-1] = data[period-1]
        
        # Calculate KAMA
        for i in range(period, len(data)):
            kama[i] = kama[i-1] + sc[i] * (data[i] - kama[i-1])
            
        return kama
    
    kama_1d = calculate_kama(close_1d, 30, 2, 30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Volume average (24-period = 12 hours for volume confirmation)
    avg_volume = np.full(n, np.nan)
    for i in range(24, n):
        avg_volume[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(24, n):
        # Skip if any required data is not ready
        if np.isnan(kama_1d_aligned[i]) or np.isnan(avg_volume[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        kama_val = kama_1d_aligned[i]
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirm = vol > 2.0 * avg_vol
        
        if position == 0:
            # Long: Price above KAMA + volume confirmation
            if price > kama_val and volume_confirm:
                position = 1
                signals[i] = position_size
            # Short: Price below KAMA + volume confirmation
            elif price < kama_val and volume_confirm:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price crosses below KAMA
            if price < kama_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Price crosses above KAMA
            if price > kama_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_KAMA_Volume_Adaptive"
timeframe = "12h"
leverage = 1.0