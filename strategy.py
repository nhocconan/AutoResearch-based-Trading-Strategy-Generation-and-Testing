#!/usr/bin/env python3
# 4h_RSI2_MeanReversion_Bollinger_Bands
# Hypothesis: Mean reversion strategy using 2-period RSI and Bollinger Bands (20,2.0) for BTC/ETH/SOL.
# Enters long when RSI(2) < 10 and price touches lower Bollinger Band; short when RSI(2) > 90 and price touches upper band.
# Exits when RSI(2) crosses above 50 (long) or below 50 (short) or price reaches middle band.
# Bollinger Band width acts as volatility filter: only trade when width > 20th percentile (avoid low volatility chop).
# Designed to work in both bull and bear markets by capturing short-term reversals.

name = "4h_RSI2_MeanReversion_Bollinger_Bands"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 2-period RSI
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)  # align with close
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    rsi_gain = np.full_like(close, np.nan)
    rsi_loss = np.full_like(close, np.nan)
    
    if len(gain) >= 2:
        rsi_gain[1] = np.mean(gain[0:2])
        rsi_loss[1] = np.mean(loss[0:2])
        for i in range(2, len(gain)):
            rsi_gain[i] = (rsi_gain[i-1] * 1 + gain[i]) / 2
            rsi_loss[i] = (rsi_loss[i-1] * 1 + loss[i]) / 2
    
    rsi = np.full_like(close, np.nan)
    valid = (~np.isnan(rsi_loss)) & (rsi_loss != 0)
    rsi[valid] = 100 - (100 / (1 + rsi_gain[valid] / rsi_loss[valid]))
    
    # Calculate Bollinger Bands (20, 2.0)
    bb_length = 20
    bb_mult = 2.0
    
    # Middle band (SMA)
    bb_mid = np.full_like(close, np.nan)
    if len(close) >= bb_length:
        bb_mid[bb_length-1] = np.mean(close[0:bb_length])
        for i in range(bb_length, len(close)):
            bb_mid[i] = (bb_mid[i-1] * (bb_length-1) + close[i]) / bb_length
    
    # Standard deviation
    bb_std = np.full_like(close, np.nan)
    if len(close) >= bb_length:
        for i in range(bb_length-1, len(close)):
            bb_std[i] = np.std(close[i-bb_length+1:i+1])
    
    # Upper and lower bands
    bb_upper = np.full_like(close, np.nan)
    bb_lower = np.full_like(close, np.nan)
    valid_bb = ~np.isnan(bb_mid) & ~np.isnan(bb_std)
    bb_upper[valid_bb] = bb_mid[valid_bb] + bb_mult * bb_std[valid_bb]
    bb_lower[valid_bb] = bb_mid[valid_bb] - bb_mult * bb_std[valid_bb]
    
    # Bollinger Band width
    bb_width = np.full_like(close, np.nan)
    bb_width[valid_bb] = (bb_upper[valid_bb] - bb_lower[valid_bb]) / bb_mid[valid_bb]
    
    # Volatility filter: BB width > 20th percentile (avoid low volatility chop)
    bb_width_valid = bb_width[~np.isnan(bb_width)]
    if len(bb_width_valid) >= 20:
        bb_width_threshold = np.percentile(bb_width_valid, 20)
        vol_filter = bb_width > bb_width_threshold
    else:
        vol_filter = np.ones_like(close, dtype=bool)  # default to true if not enough data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_length, 2)  # Need BB and RSI
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(bb_mid[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: RSI(2) < 10 and price at or below lower Bollinger Band
            if rsi[i] < 10 and close[i] <= bb_lower[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: RSI(2) > 90 and price at or above upper Bollinger Band
            elif rsi[i] > 90 and close[i] >= bb_upper[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI(2) crosses above 50 or price reaches middle band
            if rsi[i] > 50 or close[i] >= bb_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI(2) crosses below 50 or price reaches middle band
            if rsi[i] < 50 or close[i] <= bb_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals