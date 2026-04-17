#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour Bollinger Band breakout with 4-hour RSI filter and volume confirmation.
# Breakouts capture momentum moves in both bull and bear markets. The 4-hour RSI filter
# avoids counter-trend trades (long when RSI>50, short when RSI<50), while volume
# confirmation validates breakout strength. Uses 1-hour timeframe for entry timing.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data for RSI filter ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # RSI calculation (14-period) on 4h data
    def calculate_rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        # Wilder's smoothing
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_4h = calculate_rsi(close_4h, 14)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # === 1h data for Bollinger Bands ===
    # Bollinger Bands (20, 2) on 1h data
    bb_period = 20
    bb_std = 2
    
    sma = np.full_like(close, np.nan)
    std_dev = np.full_like(close, np.nan)
    
    for i in range(bb_period - 1, len(close)):
        sma[i] = np.mean(close[i - bb_period + 1:i + 1])
        std_dev[i] = np.std(close[i - bb_period + 1:i + 1])
    
    upper_band = sma + (std_dev * bb_std)
    lower_band = sma - (std_dev * bb_std)
    
    # Volume average (20-period)
    vol_avg = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_avg[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    warmup = max(50, bb_period)  # Sufficient for all indicators
    
    for i in range(warmup, n):
        if (np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or
            np.isnan(rsi_4h_aligned[i]) or
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        vol_filter = volume[i] > 1.5 * vol_avg[i]
        
        if position == 0:
            # Long: price breaks above upper band + RSI > 50 (bullish bias) + volume
            if close[i] > upper_band[i] and \
               rsi_4h_aligned[i] > 50 and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below lower band + RSI < 50 (bearish bias) + volume
            elif close[i] < lower_band[i] and \
                 rsi_4h_aligned[i] < 50 and vol_filter:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below lower band (mean reversion)
            if close[i] < lower_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price breaks above upper band (mean reversion)
            if close[i] > upper_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_BB20_4hRSI_VolumeFilter"
timeframe = "1h"
leverage = 1.0