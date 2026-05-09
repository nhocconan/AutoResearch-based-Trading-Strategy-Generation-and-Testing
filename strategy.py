#!/usr/bin/env python3
# 1d_KAMA_Trend_RSI_MeanReversion_v2
# Hypothesis: On daily timeframe, use KAMA for trend direction, RSI for mean-reversion entries, and volume confirmation.
# Long when KAMA up (bullish trend) + RSI < 30 (oversold) + volume > 1.3x average.
# Short when KAMA down (bearish trend) + RSI > 70 (overbought) + volume > 1.3x average.
# Uses weekly trend filter to avoid counter-trend trades in strong trends.
# Designed for low trade frequency (<25/year) to minimize fee drag and work in both bull and bear markets.

name = "1d_KAMA_Trend_RSI_MeanReversion_v2"
timeframe = "1d"
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
    
    # Get weekly data for trend filter (higher timeframe)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Get daily data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average ) on daily
    # Using ER (Efficiency Ratio) with 10-day fast, 30-day slow
    def calculate_kama(close, fast=10, slow=30):
        n = len(close)
        kama = np.full(n, np.nan)
        if n < slow:
            return kama
        
        # Initialize first value
        kama[slow-1] = close[slow-1]
        
        for i in range(slow, n):
            # Direction
            direction = abs(close[i] - close[i-10])
            
            # Volatility
            volatility = 0
            for j in range(i-9, i+1):
                if j > 0:
                    volatility += abs(close[j] - close[j-1])
            
            # Avoid division by zero
            if volatility == 0:
                er = 0
            else:
                er = direction / volatility
            
            # Smoothing constants
            sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
            
            # KAMA calculation
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
        
        return kama
    
    kama_1d = calculate_kama(close_1d, 10, 30)
    
    # Calculate RSI (14) on daily
    def calculate_rsi(close, period=14):
        n = len(close)
        rsi = np.full(n, np.nan)
        if n < period + 1:
            return rsi
        
        # Calculate changes
        delta = np.diff(close)
        
        # Separate gains and losses
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        # First average
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        # Avoid division by zero
        if avg_loss == 0:
            rsi[period] = 100
        else:
            rsi[period] = 100 - (100 / (1 + avg_gain / avg_loss))
        
        # Subsequent values using Wilder's smoothing
        for i in range(period + 1, n):
            avg_gain = (gains[i-1] + (period - 1) * avg_gain) / period
            avg_loss = (losses[i-1] + (period - 1) * avg_loss) / period
            
            if avg_loss == 0:
                rsi[i] = 100
            else:
                rsi[i] = 100 - (100 / (1 + avg_gain / avg_loss))
        
        return rsi
    
    rsi_1d = calculate_rsi(close_1d, 14)
    
    # Calculate weekly EMA for trend filter
    ema20_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 20:
        ema20_1w[19] = np.mean(close_1w[0:20])
        for i in range(20, len(close_1w)):
            ema20_1w[i] = (close_1w[i] * 2 + ema20_1w[i-1] * 18) / 20
    
    # Align weekly EMA to daily timeframe
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Align daily indicators to daily timeframe (no alignment needed, but using for consistency)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume filter: current volume vs 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20, 20)  # Need KAMA, RSI, weekly EMA, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trends
        kama_up = close[i] > kama_1d_aligned[i]
        weekly_up = close[i] > ema20_1w_aligned[i]
        
        if position == 0:
            # Enter long: KAMA up (bullish trend) + RSI < 30 (oversold) + volume confirmation
            # Only take long if weekly trend is also up to avoid fighting strong downtrends
            if kama_up and rsi_1d_aligned[i] < 30 and volume_ratio[i] > 1.3 and weekly_up:
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA down (bearish trend) + RSI > 70 (overbought) + volume confirmation
            # Only take short if weekly trend is also down to avoid fighting strong uptrends
            elif not kama_up and rsi_1d_aligned[i] > 70 and volume_ratio[i] > 1.3 and not weekly_up:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA turns down or RSI > 70 (overbought)
            if not kama_up or rsi_1d_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA turns up or RSI < 30 (oversold)
            if kama_up or rsi_1d_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals