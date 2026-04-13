#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze with RSI momentum and 1d volume confirmation.
# Bollinger Band squeeze (low volatility) precedes explosive moves. RSI > 55 confirms bullish momentum, < 45 bearish.
# 1d volume > 1.5x average confirms institutional interest. Works in both bull (breakouts up) and bear (breakouts down).
# Target: 20-40 trades per year (80-160 total over 4 years) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) on 4h
    bb_length = 20
    bb_mult = 2
    sma = np.full(n, np.nan)
    std = np.full(n, np.nan)
    for i in range(bb_length - 1, n):
        sma[i] = np.mean(close[i - bb_length + 1:i + 1])
        std[i] = np.std(close[i - bb_length + 1:i + 1])
    upper = sma + bb_mult * std
    lower = sma - bb_mult * std
    bb_width = (upper - lower) / sma  # normalized width
    
    # Bollinger Band squeeze: width < 20th percentile of last 50 periods
    bb_width_percentile = np.full(n, np.nan)
    for i in range(50, n):
        past_widths = bb_width[i - 50:i]
        bb_width_percentile[i] = (np.sum(past_widths < bb_width[i]) / 50) * 100
    squeeze = bb_width_percentile < 20  # squeeze when width in lowest 20%
    
    # RSI(14) for momentum
    rsi_length = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(rsi_length, n):
        if i == rsi_length:
            avg_gain[i] = np.mean(gain[i - rsi_length + 1:i + 1])
            avg_loss[i] = np.mean(loss[i - rsi_length + 1:i + 1])
        else:
            avg_gain[i] = (avg_gain[i - 1] * (rsi_length - 1) + gain[i]) / rsi_length
            avg_loss[i] = (avg_loss[i - 1] * (rsi_length - 1) + loss[i]) / rsi_length
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # 1-day data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    avg_volume_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        avg_volume_1d[i] = np.mean(volume_1d[i - 20:i])
    volume_1d_avg_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(squeeze[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_1d_avg_aligned[i])):
            signals[i] = 0.0
            continue
        
        vol_1d = volume[i]  # current 4h volume
        avg_vol_1d = volume_1d_avg_aligned[i]  # average 1d volume aligned to 4h
        
        # Volume confirmation: current 4h volume > 1.5x average 1d volume (scaled)
        # Scale: 1d volume represents ~6x 4h bars, so divide by 6 for per-bar comparison
        volume_confirm = vol_1d > 1.5 * (avg_vol_1d / 6)
        
        if position == 0:
            # Long: Squeeze + RSI > 55 (bullish momentum) + volume confirmation
            if squeeze[i] and (rsi[i] > 55) and volume_confirm:
                position = 1
                signals[i] = position_size
            # Short: Squeeze + RSI < 45 (bearish momentum) + volume confirmation
            elif squeeze[i] and (rsi[i] < 45) and volume_confirm:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Squeeze ends or RSI < 50
            if (not squeeze[i]) or (rsi[i] < 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Squeeze ends or RSI > 50
            if (not squeeze[i]) or (rsi[i] > 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_BollingerSqueeze_RSI_Momentum"
timeframe = "4h"
leverage = 1.0