#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA (Kaufman Adaptive Moving Average) trend filter + RSI(2) extreme reversal 
# with volume spike confirmation. Uses weekly timeframe to filter trades in alignment with higher timeframe trend.
# KAMA adapts to market noise - faster in trending markets, slower in ranging markets.
# RSI(2) captures short-term extremes for mean reversion entries.
# Volume spike confirms institutional participation.
# Designed for low turnover (target: 15-25 trades/year) to minimize fee drag.
# Works in both bull and bear markets by combining trend following with mean reversion.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate KAMA on weekly data
    # ER (Efficiency Ratio) = |change| / volatility
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility = np.sum(np.abs(np.diff(close_1w, prepend=close_1w[0])), axis=0) if False else None  # placeholder
    # Proper volatility calculation: sum of absolute changes over period
    volatility = np.zeros_like(close_1w)
    for i in range(1, len(close_1w)):
        volatility[i] = volatility[i-1] + np.abs(close_1w[i] - close_1w[i-1])
    # For efficiency ratio over 10 periods
    er = np.zeros_like(close_1w)
    for i in range(10, len(close_1w)):
        price_change = np.abs(close_1w[i] - close_1w[i-10])
        sum_abs_change = 0
        for j in range(1, 11):
            sum_abs_change += np.abs(close_1w[i-j+1] - close_1w[i-j])
        if sum_abs_change > 0:
            er[i] = price_change / sum_abs_change
        else:
            er[i] = 0
    # Smoothing constants
    sc = (er * (0.66 - 0.06) + 0.06) ** 2
    # Calculate KAMA
    kama = np.full_like(close_1w, np.nan)
    kama[9] = close_1w[9]  # seed
    for i in range(10, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # Align KAMA to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # Calculate daily RSI(2)
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[1] = gain[1]
    avg_loss[1] = loss[1]
    for i in range(2, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 1 + gain[i]) / 2
        avg_loss[i] = (avg_loss[i-1] * 1 + loss[i]) / 2
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 20-day average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi[i]
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long signal: price above KAMA (uptrend) + RSI(2) oversold + volume spike
            if price > kama_val and rsi_val < 15 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short signal: price below KAMA (downtrend) + RSI(2) overbought + volume spike
            elif price < kama_val and rsi_val > 85 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on RSI(2) overbought or price below KAMA
                if rsi_val > 85 or price < kama_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on RSI(2) oversold or price above KAMA
                if rsi_val < 15 or price > kama_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_KAMA_RSI2_VolumeSpike"
timeframe = "1d"
leverage = 1.0