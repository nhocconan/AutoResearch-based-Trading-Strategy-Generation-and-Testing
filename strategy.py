#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h RSI + Volume + Volume Spike filter for mean reversion.
# RSI(14) < 30 for long, > 70 for short, with volume > 1.5x 20-period average and volume spike > 2x average volume.
# Uses 1-day RSI for confirmation to avoid false signals in strong trends.
# Target: 20-30 trades/year per symbol to stay within frequency limits.
name = "12h_RSI_Volume_Spike_MeanReversion"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate RSI (14-period)
    def calculate_rsi(data, period=14):
        if len(data) < period + 1:
            return np.full_like(data, np.nan)
        delta = np.diff(data)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(data)
        avg_loss = np.zeros_like(data)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period + 1, len(data)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
        rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1d = calculate_rsi(close_1d, 14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Get 12h average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ma_5 = pd.Series(volume).rolling(window=5, min_periods=5).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure RSI (14*2+2) and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(vol_ma_5[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        vol_ma_short = vol_ma_5[i]
        vol = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = vol > 1.5 * vol_ma
        # Volume spike: current volume > 2x 5-period average
        volume_spike = vol > 2.0 * vol_ma_short
        
        if position == 0:
            # Enter long when RSI oversold with volume confirmation and spike
            if rsi_val < 30 and volume_confirmed and volume_spike:
                signals[i] = 0.25
                position = 1
            # Enter short when RSI overbought with volume confirmation and spike
            elif rsi_val > 70 and volume_confirmed and volume_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when RSI returns to neutral (50) or overbought
            if rsi_val >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when RSI returns to neutral (50) or oversold
            if rsi_val <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals