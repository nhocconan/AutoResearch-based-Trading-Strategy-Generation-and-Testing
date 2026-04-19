#!/usr/bin/env python3
# 12h_Camarilla_Touch_Reversal
# Hypothesis: Price often reverses at Camarilla S3/R3 levels with volume exhaustion and overbought/oversold RSI.
# Uses mean reversion at strong support/resistance in ranging markets (Chop > 61.8).
# Works in bull/bear via regime filter - avoids trending markets (Chop < 38.2).
# Target: 50-150 total trades over 4 years by requiring confluence of 3+ conditions.

name = "12h_Camarilla_Touch_Reversal"
timeframe = "12h"
leverage = 1.0

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
    
    # Calculate RSI(14) for overbought/oversold
    def calculate_rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        # Wilder's smoothing
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Calculate Choppiness Index
    def calculate_chop(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # Sum of True Range over period
        tr_sum = np.zeros_like(close)
        for i in range(len(close)):
            if i < period:
                tr_sum[i] = np.nan
            else:
                tr_sum[i] = np.nansum(tr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        max_high = np.zeros_like(high)
        min_low = np.zeros_like(low)
        for i in range(len(close)):
            if i < period:
                max_high[i] = np.nan
                min_low[i] = np.nan
            else:
                max_high[i] = np.nanmax(high[i-period+1:i+1])
                min_low[i] = np.nanmin(low[i-period+1:i+1])
        
        # Chop calculation
        chop = np.full_like(close, np.nan)
        for i in range(len(close)):
            if (not np.isnan(tr_sum[i]) and tr_sum[i] > 0 and 
                not np.isnan(max_high[i]) and not np.isnan(min_low[i]) and
                max_high[i] != min_low[i]):
                chop[i] = 100 * np.log10(tr_sum[i] / (max_high[i] - min_low[i])) / np.log10(period)
        return chop
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    ph = df_1d['high'].shift(1).values
    pl = df_1d['low'].shift(1).values
    pc = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    rang = ph - pl
    # Avoid division by zero in case of zero range
    rang = np.where(rang == 0, 1e-10, rang)
    
    # S3 and R3 levels (more extreme than S1/R1)
    s3 = pc - (rang * 1.1 / 6)
    r3 = pc + (rang * 1.1 / 6)
    
    # Align to 12h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    
    # Calculate indicators on 12h data
    rsi = calculate_rsi(close, 14)
    chop = calculate_chop(high, low, close, 14)
    
    # Volume exhaustion: current volume < 0.7 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_exhaustion = volume < (volume_ma * 0.7)
    
    signals = np.zeros(n)
    
    # Start after enough data for all indicators
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in ranging markets (Chop > 61.8)
        ranging_market = chop[i] > 61.8
        
        if ranging_market:
            # Long setup: price at or below S3 with RSI oversold and volume exhaustion
            if (low[i] <= s3_aligned[i] and 
                rsi[i] < 30 and 
                volume_exhaustion[i]):
                signals[i] = 0.25
            # Short setup: price at or above R3 with RSI overbought and volume exhaustion
            elif (high[i] >= r3_aligned[i] and 
                  rsi[i] > 70 and 
                  volume_exhaustion[i]):
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:
            # In trending markets, stay flat
            signals[i] = 0.0
    
    return signals