#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d < 50):
        return np.zeros(n)
    
    # Calculate 1d RSI (14-period)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1d Bollinger Bands (20, 2.0)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper = sma_20 + 2 * std_20
    lower = sma_20 - 2 * std_20
    
    # Calculate 12h price position within Bollinger Bands
    close_12h = prices['close'].values
    bb_position = np.zeros_like(close_12h)
    bb_width = upper - lower
    bb_position = np.where(bb_width != 0, (close_12h - lower) / bb_width, 0.5)
    
    # Align 1d indicators to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    bb_position_aligned = align_htf_to_ltf(prices, df_1d, bb_position)
    
    # Calculate 12h volume average (20-period)
    vol_12h = prices['volume'].values
    vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Bollinger warmup
        # Skip if data not ready
        if (np.isnan(rsi_aligned[i]) or np.isnan(bb_position_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume (12h)
        price_close = prices['close'].iloc[i]
        vol_current = vol_12h[i]
        
        if position == 0:
            # Enter long: Price at lower BB + RSI oversold + volume confirmation
            if (bb_position_aligned[i] <= 0.2 and
                rsi_aligned[i] < 30 and
                vol_current > 1.2 * vol_ma_20[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Price at upper BB + RSI overbought + volume confirmation
            elif (bb_position_aligned[i] >= 0.8 and
                  rsi_aligned[i] > 70 and
                  vol_current > 1.2 * vol_ma_20[i]):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: Price returns to middle of BB or RSI normalizes
            exit_signal = False
            
            if position == 1:
                # Exit long: Price reaches middle BB or RSI > 50
                if (bb_position_aligned[i] >= 0.5 or
                    rsi_aligned[i] > 50):
                    exit_signal = True
            elif position == -1:
                # Exit short: Price reaches middle BB or RSI < 50
                if (bb_position_aligned[i] <= 0.5 or
                    rsi_aligned[i] < 50):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_BB_RSI_MeanReversion_Volume"
timeframe = "12h"
leverage = 1.0