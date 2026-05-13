#!/usr/bin/env python3
"""
4h_KAMA_Direction_RSI_Chop_Filter_v1
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) to determine trend direction (price > KAMA = up, price < KAMA = down) on 4h timeframe. Enter long when RSI crosses above 50 in uptrend, short when RSI crosses below 50 in downtrend, with volume confirmation (volume > 1.5x 20-period average) and chop filter (Choppiness Index > 61.8 for ranging markets). Exit when RSI crosses back below 50 (long) or above 50 (short). Designed to capture momentum shifts with reduced whipsaw in both bull and bear markets.
"""

name = "4h_KAMA_Direction_RSI_Chop_Filter_v1"
timeframe = "4h"
leverage = 1.0

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
    
    # KAMA calculation
    def kama(close, er_len=10, fast_len=2, slow_len=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) == 1 else np.convolve(np.abs(np.diff(close)), np.ones(er_len), 'same')
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast_len+1) - 2/(slow_len+1)) + 2/(slow_len+1)) ** 2
        kama_out = np.zeros_like(close)
        kama_out[0] = close[0]
        for i in range(1, len(close)):
            kama_out[i] = kama_out[i-1] + sc[i] * (close[i] - kama_out[i-1])
        return kama_out
    
    kama_vals = kama(close, 10, 2, 30)
    
    # RSI calculation
    def rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[length] = np.mean(gain[1:length+1])
        avg_loss[length] = np.mean(loss[1:length+1])
        for i in range(length+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i]) / length
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_out = 100 - (100 / (1 + rs))
        return rsi_out
    
    rsi_vals = rsi(close, 14)
    
    # Choppiness Index calculation
    def chop(high, low, close, length=14):
        atr = np.zeros_like(close)
        tr1 = np.abs(high - low)
        tr2 = np.abs(np.roll(high, 1) - close)
        tr3 = np.abs(np.roll(low, 1) - close)
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.convolve(tr, np.ones(length), 'full')[:len(close)] / length
        atr[:length-1] = np.nan
        
        highest_high = np.zeros_like(close)
        lowest_low = np.zeros_like(close)
        for i in range(length-1, len(close)):
            highest_high[i] = np.max(high[i-length+1:i+1])
            lowest_low[i] = np.min(low[i-length+1:i+1])
        
        sum_atr = np.nansum(atr[-length:]) if len(atr) >= length else np.nansum(atr)
        range_hl = highest_high - lowest_low
        chop_vals = 100 * np.log10(sum_atr / range_hl) / np.log10(length)
        return chop_vals
    
    chop_vals = chop(high, low, close, 14)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.convolve(volume, np.ones(20)/20, 'same')
    vol_ma[:19] = np.nan
    volume_confirmed = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup period
        if np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or np.isnan(chop_vals[i]) or np.isnan(volume_confirmed[i]):
            signals[i] = 0.0
            continue
            
        # Only trade in ranging markets (Chop > 61.8) to avoid whipsaw in strong trends
        if chop_vals[i] <= 61.8:
            signals[i] = 0.0
            continue
            
        if position == 0:
            # LONG: Price above KAMA, RSI crosses above 50, volume confirmation
            if close[i] > kama_vals[i] and rsi_vals[i] > 50 and rsi_vals[i-1] <= 50 and volume_confirmed[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA, RSI crosses below 50, volume confirmation
            elif close[i] < kama_vals[i] and rsi_vals[i] < 50 and rsi_vals[i-1] >= 50 and volume_confirmed[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI crosses back below 50
            if rsi_vals[i] < 50 and rsi_vals[i-1] >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI crosses back above 50
            if rsi_vals[i] > 50 and rsi_vals[i-1] <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals