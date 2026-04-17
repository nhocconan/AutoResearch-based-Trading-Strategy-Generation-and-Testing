# 4h_KAMA_Direction_RSI_ChopFilter_V1
# KAMA direction + RSI + chop filter on 4h timeframe
# KAMA adapts to market efficiency, RSI provides momentum, chop filter identifies trending vs ranging markets
# Designed to work in both bull and bear markets by adapting to market conditions

#!/usr/bin/env python3
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
    
    # === 1d KAMA (10-period ER, 2 and 30 SC) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i >= 10:
            change_sum = np.sum(change[i-9:i+1])
            volatility_sum = np.sum(volatility[i-9:i+1])
            er[i] = change_sum / volatility_sum if volatility_sum != 0 else 0
        else:
            er[i] = 0
    
    # Smoothing constants
    sc_fast = 2 / (2 + 1)  # EMA(2)
    sc_slow = 2 / (30 + 1)  # EMA(30)
    sc = (er * (sc_fast - sc_slow) + sc_slow) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # === 1d RSI(14) ===
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    
    # First average
    if len(close_1d) >= 14:
        avg_gain[13] = np.mean(gain[1:15])
        avg_loss[13] = np.mean(loss[1:15])
    
    # Subsequent averages
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)
    rsi = np.where(avg_gain == 0, 0, rsi)
    
    # === 1d Choppiness Index (14-period) ===
    atr_14 = np.zeros_like(close_1d)
    tr = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close_1d[i-1]),
            abs(low[i] - close_1d[i-1])
        )
    # First TR
    tr[0] = high[0] - low[0]
    
    # Calculate ATR
    for i in range(len(atr_14)):
        if i < 14:
            atr_14[i] = np.mean(tr[1:i+1]) if i > 0 else tr[0]
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = np.zeros_like(close_1d)
    lowest_low = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i >= 13:
            highest_high[i] = np.max(high[i-13:i+1])
            lowest_low[i] = np.min(low[i-13:i+1])
        else:
            highest_high[i] = np.max(high[0:i+1])
            lowest_low[i] = np.min(low[0:i+1])
    
    # Chop calculation
    chop = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if atr_14[i] > 0 and i >= 13:
            sum_tr = np.sum(tr[i-13:i+1])
            chop[i] = 100 * np.log10(sum_tr / (atr_14[i] * 14)) / np.log10(14)
        else:
            chop[i] = 50  # neutral
    
    # === Align indicators to 4h timeframe ===
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 4h Volume confirmation ===
    # Calculate 20-period average volume
    vol_ma_20 = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_confirm = volume > vol_ma_20 * 1.5
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Price above KAMA AND RSI > 50 AND Chop < 61.8 (trending) AND Volume confirmation
            if (close[i] > kama_aligned[i] and 
                rsi_aligned[i] > 50 and 
                chop_aligned[i] < 61.8 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: Price below KAMA AND RSI < 50 AND Chop < 61.8 (trending) AND Volume confirmation
            elif (close[i] < kama_aligned[i] and 
                  rsi_aligned[i] < 50 and 
                  chop_aligned[i] < 61.8 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: Price below KAMA OR RSI < 40 OR Chop > 61.8 (choppy)
            if (close[i] < kama_aligned[i] or 
                rsi_aligned[i] < 40 or 
                chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price above KAMA OR RSI > 60 OR Chop > 61.8 (choppy)
            if (close[i] > kama_aligned[i] or 
                rsi_aligned[i] > 60 or 
                chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Direction_RSI_ChopFilter_V1"
timeframe = "4h"
leverage = 1.0