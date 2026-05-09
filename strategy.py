#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_KAMA_Trend_RSI25_75_Range_Filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # KAMA parameters
    er_length = 10
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close, n=er_length))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if hasattr(np.sum, 'axis') else np.sum(np.abs(np.diff(close)))
    # More efficient calculation
    er = np.zeros_like(close)
    for i in range(er_length, len(close)):
        if i - er_length >= 0:
            price_change = np.abs(close[i] - close[i - er_length])
            volatility_sum = np.sum(np.abs(np.diff(close[i - er_length + 1:i + 1])))
            if volatility_sum > 0:
                er[i] = price_change / volatility_sum
            else:
                er[i] = 0
    
    # Calculate Smoothing Constant and KAMA
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Align KAMA to daily (no additional delay needed for KAMA itself)
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # RSI(14) for range/extreme detection
    rsi_length = 14
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[rsi_length] = np.mean(gain[1:rsi_length+1])
    avg_loss[rsi_length] = np.mean(loss[1:rsi_length+1])
    
    for i in range(rsi_length + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_length - 1) + gain[i]) / rsi_length
        avg_loss[i] = (avg_loss[i-1] * (rsi_length - 1) + loss[i]) / rsi_length
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:rsi_length+1] = np.nan  # Mark initial values as NaN
    
    # Bollinger Bands for volatility/chop regime (20, 2)
    bb_length = 20
    bb_mult = 2
    sma = np.full_like(close, np.nan)
    bb_std = np.full_like(close, np.nan)
    
    for i in range(bb_length, len(close)):
        sma[i] = np.mean(close[i-bb_length+1:i+1])
        bb_std[i] = np.std(close[i-bb_length+1:i+1])
    
    upper_band = sma + bb_mult * bb_std
    lower_band = sma - bb_mult * bb_std
    
    # Calculate Bollinger Band Width for chop detection
    bb_width = (upper_band - lower_band) / sma
    
    # Bollinger Band Width percentile for regime (252-day lookback ~1 year)
    bb_width_percentile = np.full_like(close, np.nan)
    lookback = 252
    for i in range(lookback, len(close)):
        if i - lookback >= 0:
            window = bb_width[i-lookback+1:i+1]
            valid_vals = window[~np.isnan(window)]
            if len(valid_vals) > 0:
                percentile = np.sum(valid_vals <= bb_width[i]) / len(valid_vals) * 100
                bb_width_percentile[i] = percentile
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, er_length, rsi_length+1, bb_length, lookback)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(bb_width_percentile[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Range condition: BBWidth percentile < 40 (low volatility = range)
        is_range = bb_width_percentile[i] < 40
        
        if position == 0:
            # Long: Price > KAMA AND RSI < 25 (oversold) AND in range
            if close[i] > kama_aligned[i] and rsi[i] < 25 and is_range:
                signals[i] = 0.25
                position = 1
            # Short: Price < KAMA AND RSI > 75 (overbought) AND in range
            elif close[i] < kama_aligned[i] and rsi[i] > 75 and is_range:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price < KAMA OR RSI > 50 (exiting oversold)
            if close[i] < kama_aligned[i] or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price > KAMA OR RSI < 50 (exiting overbought)
            if close[i] > kama_aligned[i] or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals