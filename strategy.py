# 1d_RSI50_Cross_With_Volume_Spike
# Hypothesis: RSI crossing above 50 with volume spike indicates bullish momentum, and below 50 indicates bearish momentum.
# Uses volume > 1.5x 20-period average for confirmation. Designed for low trade frequency (10-25/year) on 1d timeframe.
# Works in bull/bear by following momentum direction. RSI(14) avoids whipsaw vs shorter periods.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(14) calculation
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close, np.nan)
        avg_loss = np.full_like(close, np.nan)
        
        if len(close) >= period + 1:
            # Initial average
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
            
            # Wilder smoothing
            for i in range(period + 1, len(close)):
                avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
                avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
        
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
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if required data not available
        if np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: RSI crosses above 50 with volume
            if rsi[i] > 50 and rsi[i-1] <= 50 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: RSI crosses below 50 with volume
            elif rsi[i] < 50 and rsi[i-1] >= 50 and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI crosses below 50
            if rsi[i] < 50 and rsi[i-1] >= 50:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI crosses above 50
            if rsi[i] > 50 and rsi[i-1] <= 50:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_RSI50_Cross_With_Volume_Spike"
timeframe = "1d"
leverage = 1.0