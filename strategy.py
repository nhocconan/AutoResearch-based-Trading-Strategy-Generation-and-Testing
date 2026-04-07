#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily (1d) KAMA + RSI + Chop filter strategy
# Uses Kaufman Adaptive Moving Average (KAMA) for trend direction,
# RSI(14) for momentum confirmation, and Choppiness Index for regime filtering.
# Designed for low-frequency trading (1d timeframe) to minimize fee drag.
# Works in both bull and bear markets via adaptive trend following and regime awareness.

name = "daily_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) - trend direction
    # Using 10-period ER (Efficiency Ratio) and 2/30 smoothing constants
    change = np.abs(close - np.roll(close, 10))
    volatility = np.sum(np.abs(np.diff(close, axis=0)), axis=0) if len(close) > 1 else 0
    # Handle first element for volatility calculation
    vol_temp = np.abs(np.diff(close))
    volatility = np.concatenate([[vol_temp[0]] if len(vol_temp) > 0 else [0], vol_temp])
    volatility_sum = np.convolve(volatility, np.ones(10), 'valid')
    volatility_sum = np.concatenate([np.full(9, np.nan), volatility_sum]) if len(volatility_sum) < len(close) else volatility_sum
    er = np.where(volatility_sum != 0, change / volatility_sum, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # smoothing constant
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # initialize
    for i in range(10, n):
        if not np.isnan(kama[i-1]) and not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14) for momentum confirmation
    delta = np.diff(close)
    delta = np.concatenate([[0], delta])  # align length
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index for regime filtering
    # Higher values indicate ranging market, lower values indicate trending
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).sum().values
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    chop = np.where((highest_high - lowest_low) != 0, 
                    100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(atr_period), 
                    50)
    
    # Weekly trend filter (1w timeframe)
    df_1w = get_htf_data(prices, '1w')
    if len(den_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(sma_50_1w_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price crosses below KAMA or RSI < 40 (exit long)
            if close[i] < kama[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above KAMA or RSI > 60 (exit short)
            if close[i] > kama[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Determine market regime using Choppiness Index
            # CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow)
            # We'll use trend-following in trending markets, mean-reversion in ranging
            trending_market = chop[i] < 38.2
            ranging_market = chop[i] > 61.8
            
            # Weekly trend filter
            weekly_uptrend = close[i] > sma_50_1w_aligned[i]
            weekly_downtrend = close[i] < sma_50_1w_aligned[i]
            
            # Long conditions
            long_signal = False
            if trending_market and weekly_uptrend:
                # Trend following: buy when price > KAMA and RSI > 50
                if close[i] > kama[i] and rsi[i] > 50:
                    long_signal = True
            elif ranging_market:
                # Mean reversion: buy when RSI < 30 (oversold)
                if rsi[i] < 30:
                    long_signal = True
            
            # Short conditions
            short_signal = False
            if trending_market and weekly_downtrend:
                # Trend following: sell when price < KAMA and RSI < 50
                if close[i] < kama[i] and rsi[i] < 50:
                    short_signal = True
            elif ranging_market:
                # Mean reversion: sell when RSI > 70 (overbought)
                if rsi[i] > 70:
                    short_signal = True
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
    
    return signals