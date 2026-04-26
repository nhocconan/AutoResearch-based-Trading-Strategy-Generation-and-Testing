#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter_v1
Hypothesis: On 1d timeframe, KAMA trend direction combined with RSI extremes and choppiness regime filter produces high-probability mean-reversion trades in ranging markets and trend-following trades in trending markets. The strategy adapts to market regime using choppiness index: CHOP > 61.8 = range (RSI mean reversion), CHOP < 38.2 = trend (KAMA breakout). Volume confirmation reduces false signals. Target: 30-100 total trades over 4 years (7-25/year).
"""

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
    
    # === KAMA Calculation (trend direction) ===
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    vol = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # 10-period sum of absolute changes
    # Handle first 10 values
    change = np.concatenate([[np.nan]*10, change])
    vol = np.concatenate([[np.nan]*10, vol])
    er = np.where(vol != 0, change / vol, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # KAMA
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]) or np.isnan(kama[i-1]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI Calculation (14-period) ===
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    avg_gain[14] = np.nanmean(gain[1:15])
    avg_loss[14] = np.nanmean(loss[1:15])
    
    for i in range(15, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Choppiness Index (14-period) ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[np.nan], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[np.nan], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of TR over 14 periods
    tr_sum = np.full(n, np.nan)
    for i in range(14, n):
        tr_sum[i] = np.nansum(tr[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    max_high = np.full(n, np.nan)
    min_low = np.full(n, np.nan)
    for i in range(14, n):
        max_high[i] = np.nanmax(high[i-13:i+1])
        min_low[i] = np.nanmin(low[i-13:i+1])
    
    # Chop = 100 * log10(sum(tr14) / (max_high - min_low)) / log10(14)
    chop = np.full(n, np.nan)
    for i in range(14, n):
        if max_high[i] > min_low[i] and tr_sum[i] > 0:
            chop[i] = 100 * np.log10(tr_sum[i] / (max_high[i] - min_low[i])) / np.log10(14)
        else:
            chop[i] = 50  # neutral when invalid
    
    # === Volume Confirmation (20-period average) ===
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.nanmean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 30 for KAMA, 14 for RSI/Chop, 20 for volume)
    start_idx = max(30, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume confirmation
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Regime filters
        chop_high = chop[i] > 61.8  # ranging market
        chop_low = chop[i] < 38.2   # trending market
        
        # KAMA trend
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI extremes
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_neutral = 40 <= rsi[i] <= 60
        
        # Long logic
        long_signal = False
        # In ranging market: mean reversion from oversold
        if chop_high and rsi_oversold and price_above_kama and volume_spike:
            long_signal = True
        # In trending market: breakout above KAMA
        elif chop_low and price_above_kama and rsi_neutral and volume_spike:
            long_signal = True
            
        # Short logic
        short_signal = False
        # In ranging market: mean reversion from overbought
        if chop_high and rsi_overbought and price_below_kama and volume_spike:
            short_signal = True
        # In trending market: breakout below KAMA
        elif chop_low and price_below_kama and rsi_neutral and volume_spike:
            short_signal = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        # Exit long: RSI > 50 or chop regime change or volume dry up
        if rsi[i] > 50 or chop[i] < 50 or volume[i] < 0.8 * vol_ma_20[i]:
            exit_long = True
            
        # Exit short: RSI < 50 or chop regime change or volume dry up
        if rsi[i] < 50 or chop[i] < 50 or volume[i] < 0.8 * vol_ma_20[i]:
            exit_short = True
        
        # Update signals
        if long_signal and position != 1:
            signals[i] = 0.25
            position = 1
        elif short_signal and position != -1:
            signals[i] = -0.25
            position = -1
        elif exit_long and position == 1:
            signals[i] = 0.0
            position = 0
        elif exit_short and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0