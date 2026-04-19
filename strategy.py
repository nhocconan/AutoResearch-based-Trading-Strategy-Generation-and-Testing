#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion with 4h RSI and volume filter for timing.
# In both bull and bear markets, RSI extremes on 4h combined with 1h volume spikes
# offer mean-reversion opportunities. 4h RSI > 70 or < 30 indicates overextended
# moves; 1h volume spike confirms exhaustion. Enter opposite direction with tight stop.
# Target: 15-35 trades/year per symbol.
name = "1h_RSI40_VolumeSpike_MeanRev"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for RSI
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate RSI on 4h
    def calculate_rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_4h = calculate_rsi(close_4h, 14)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # 1h volume spike: current volume > 2.0 x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1h RSI for entry confirmation (avoid extremes)
    def calculate_rsi_series(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1h = calculate_rsi_series(close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(rsi_1h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_4h_val = rsi_4h_aligned[i]
        rsi_1h_val = rsi_1h[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_spike = vol > 2.0 * vol_ma
        
        # 4h RSI extremes
        rsi_overbought = rsi_4h_val > 70
        rsi_oversold = rsi_4h_val < 30
        
        if position == 0:
            # Enter short on 4h overbought + volume spike, avoid 1h overbought
            if rsi_overbought and volume_spike and rsi_1h_val < 70:
                signals[i] = -0.20
                position = -1
            # Enter long on 4h oversold + volume spike, avoid 1h oversold
            elif rsi_oversold and volume_spike and rsi_1h_val > 30:
                signals[i] = 0.20
                position = 1
        
        elif position == 1:
            # Exit long when 4h RSI returns to neutral or 1h RSI overbought
            if rsi_4h_val >= 50 or rsi_1h_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short when 4h RSI returns to neutral or 1h RSI oversold
            if rsi_4h_val <= 50 or rsi_1h_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals