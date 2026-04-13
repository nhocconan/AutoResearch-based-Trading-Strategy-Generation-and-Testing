#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d Bollinger Bands squeeze and RSI mean reversion.
# Long: Bollinger Bands width at 20-period low + RSI < 30 + price below lower BB.
# Short: Bollinger Bands width at 20-period low + RSI > 70 + price above upper BB.
# Uses 1d Bollinger Bands for volatility contraction and RSI for mean reversion signals.
# Bollinger squeeze indicates low volatility, often preceding mean reversion moves.
# RSI extremes provide entry signals in range-bound markets.
# Time filter: 00-23 UTC (all hours) to maximize opportunities while maintaining discipline.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Bollinger Bands and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Bollinger Bands (20, 2)
    bb_length = 20
    bb_mult = 2.0
    bb_basis = np.full(len(close_1d), np.nan)
    bb_dev = np.full(len(close_1d), np.nan)
    bb_upper = np.full(len(close_1d), np.nan)
    bb_lower = np.full(len(close_1d), np.nan)
    bb_width = np.full(len(close_1d), np.nan)
    
    for i in range(bb_length, len(close_1d)):
        bb_basis[i] = np.mean(close_1d[i-bb_length:i])
        bb_dev[i] = bb_mult * np.std(close_1d[i-bb_length:i])
        bb_upper[i] = bb_basis[i] + bb_dev[i]
        bb_lower[i] = bb_basis[i] - bb_dev[i]
        bb_width[i] = bb_upper[i] - bb_lower[i]
    
    # Calculate RSI (14)
    rsi_length = 14
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close_1d), np.nan)
    avg_loss = np.full(len(close_1d), np.nan)
    for i in range(rsi_length, len(close_1d)):
        if i == rsi_length:
            avg_gain[i] = np.mean(gain[i-rsi_length:i])
            avg_loss[i] = np.mean(loss[i-rsi_length:i])
        else:
            avg_gain[i] = (avg_gain[i-1] * (rsi_length-1) + gain[i]) / rsi_length
            avg_loss[i] = (avg_loss[i-1] * (rsi_length-1) + loss[i]) / rsi_length
    
    rsi = np.full(len(close_1d), np.nan)
    for i in range(rsi_length, len(close_1d)):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
        else:
            rsi[i] = 100
    
    # Bollinger Bands squeeze: width at 20-period low
    bb_width_ma = np.full(len(close_1d), np.nan)
    for i in range(20, len(close_1d)):
        bb_width_ma[i] = np.mean(bb_width[i-20:i])
    
    bb_squeeze = np.full(len(close_1d), np.nan)
    for i in range(20, len(close_1d)):
        bb_squeeze[i] = bb_width[i] < bb_width_ma[i] * 0.8  # 20% below average width
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 1d indicators to 12h
    bb_squeeze_aligned = align_htf_to_ltf(prices, df_1d, bb_squeeze)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(bb_squeeze_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        squeeze = bb_squeeze_aligned[i]
        rsi_val = rsi_aligned[i]
        upper_bb = bb_upper_aligned[i]
        lower_bb = bb_lower_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        if position == 0:
            # Long: Bollinger squeeze + RSI < 30 + price below lower BB + volume
            if (squeeze and rsi_val < 30 and price < lower_bb and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Bollinger squeeze + RSI > 70 + price above upper BB + volume
            elif (squeeze and rsi_val > 70 and price > upper_bb and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses above middle BB or RSI > 70
            bb_middle_aligned = align_htf_to_ltf(prices, df_1d, np.full(len(close_1d), np.nan))
            for j in range(bb_length, len(close_1d)):
                bb_middle_aligned[j] = np.mean(close_1d[j-bb_length:j])
            bb_middle = bb_middle_aligned[i] if not np.isnan(bb_middle_aligned[i]) else (upper_bb + lower_bb) / 2
            
            if price > bb_middle or rsi_val > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses below middle BB or RSI < 30
            bb_middle_aligned = align_htf_to_ltf(prices, df_1d, np.full(len(close_1d), np.nan))
            for j in range(bb_length, len(close_1d)):
                bb_middle_aligned[j] = np.mean(close_1d[j-bb_length:j])
            bb_middle = bb_middle_aligned[i] if not np.isnan(bb_middle_aligned[i]) else (upper_bb + lower_bb) / 2
            
            if price < bb_middle or rsi_val < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Bollinger_Squeeze_RSI"
timeframe = "12h"
leverage = 1.0