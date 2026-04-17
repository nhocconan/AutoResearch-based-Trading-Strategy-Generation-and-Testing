#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Weekly High/Low (from weekly data) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly high and low (weekly candle values)
    # For weekly: we need the highest high and lowest low of the completed weekly candle
    weekly_high = high_1w  # each value is the high of that weekly candle
    weekly_low = low_1w    # each value is the low of that weekly candle
    
    # Align weekly high/low to daily timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # === 1d RSI (14-period) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    period = 14
    for i in range(len(gain)):
        if i < period:
            if i == 0:
                avg_gain[i] = gain[i]
                avg_loss[i] = loss[i]
            else:
                avg_gain[i] = (avg_gain[i-1] * (i-1) + gain[i]) / i
                avg_loss[i] = (avg_loss[i-1] * (i-1) + loss[i]) / i
        else:
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss == 0] = 100
    
    # === 1d Average True Range (14-period) ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = np.full_like(tr, np.nan)
    for i in range(len(tr)):
        if i < 14:
            if i == 0:
                atr[i] = tr[i]
            else:
                atr[i] = (atr[i-1] * i + tr[i]) / (i + 1)
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # === Volume Spike Detection ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    volume_spike = volume > vol_ma_20 * 2.0  # Volume > 2x 20-period average
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_high_aligned[i]) or 
            np.isnan(weekly_low_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above weekly high + RSI < 50 (not overbought) + volume spike
            if (close[i] > weekly_high_aligned[i] and 
                rsi[i] < 50 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below weekly low + RSI > 50 (not oversold) + volume spike
            elif (close[i] < weekly_low_aligned[i] and 
                  rsi[i] > 50 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price closes below weekly low OR RSI > 70 (overbought)
            if (close[i] < weekly_low_aligned[i] or 
                rsi[i] > 70):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes above weekly high OR RSI < 30 (oversold)
            if (close[i] > weekly_high_aligned[i] or 
                rsi[i] < 30):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyBreakout_RSI_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0