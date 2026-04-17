#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Elder Ray Index with 1-day ATR filter and volume confirmation.
# Elder Ray measures bull/bear power relative to EMA. In bull markets, bull power > 0 and rising indicates strength.
# In bear markets, bear power < 0 and falling indicates weakness. Combined with volume confirmation and ATR filter
# to avoid false signals, this should capture sustained momentum while filtering weak moves.
# Target: 20-40 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for ATR filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ATR calculation on daily data (14-period)
    def calculate_atr(high, low, close, period=14):
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(high)
        if len(high) > period:
            atr[period] = np.mean(tr[1:period+1])
            for i in range(period+1, len(high)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 1d data for EMA (Elder Ray base) ===
    ema_period = 13
    ema_1d = np.zeros_like(close_1d)
    if len(close_1d) > 0:
        ema_1d[0] = close_1d[0]
        for i in range(1, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 / (ema_period + 1)) + (ema_1d[i-1] * (ema_period - 1) / (ema_period + 1))
    
    # Elder Ray components
    bull_power_1d = high_1d - ema_1d
    bear_power_1d = low_1d - ema_1d
    
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # === 4h data for volume confirmation ===
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    
    # Volume average (20-period)
    vol_avg20_4h = np.zeros_like(volume_4h)
    for i in range(len(volume_4h)):
        if i >= 19:
            vol_avg20_4h[i] = np.mean(volume_4h[i-19:i+1])
        else:
            vol_avg20_4h[i] = np.nan
    
    vol_avg20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_avg20_4h)
    
    signals = np.zeros(n)
    position = 0
    warmup = 100  # Sufficient for all indicators
    
    for i in range(warmup, n):
        if (np.isnan(atr_1d_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or
            np.isnan(vol_avg20_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        vol_4h_current = align_htf_to_ltf(prices, df_4h, volume_4h)[i]
        vol_filter = vol_4h_current > 1.5 * vol_avg20_4h_aligned[i]
        
        if position == 0:
            # Long: bull power positive and rising + sufficient volatility + volume
            if bull_power_aligned[i] > 0 and bull_power_aligned[i] > bull_power_aligned[i-1] and \
               atr_1d_aligned[i] > 0.005 * close[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: bear power negative and falling + sufficient volatility + volume
            elif bear_power_aligned[i] < 0 and bear_power_aligned[i] < bear_power_aligned[i-1] and \
                 atr_1d_aligned[i] > 0.005 * close[i] and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bull power turns negative
            if bull_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bear power turns positive
            if bear_power_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_ElderRay_1dATR_VolumeFilter"
timeframe = "4h"
leverage = 1.0