#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour MACD histogram combined with 1-day RSI mean-reversion and 1-week volume confirmation.
# MACD provides momentum direction, RSI identifies overbought/oversold for mean-reversion entries,
# Volume confirms institutional participation. Designed for 12h timeframe to target 50-150 trades over 4 years.
# Works in bull markets via MACD momentum and in bear markets via RSI mean-reversion at extremes.

name = "12h_macd1d_rsi1w_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day MACD(12,26,9) for momentum
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # EMA calculations
    ema12 = np.full(len(close_1d), np.nan)
    ema26 = np.full(len(close_1d), np.nan)
    
    if len(close_1d) >= 12:
        ema12[11] = np.mean(close_1d[0:12])
        for i in range(12, len(close_1d)):
            ema12[i] = (close_1d[i] * 2 / 13) + (ema12[i-1] * 11 / 13)
    
    if len(close_1d) >= 26:
        ema26[25] = np.mean(close_1d[0:26])
        for i in range(26, len(close_1d)):
            ema26[i] = (close_1d[i] * 2 / 27) + (ema26[i-1] * 25 / 27)
    
    # MACD line
    macd_line = np.full(len(close_1d), np.nan)
    for i in range(len(close_1d)):
        if not np.isnan(ema12[i]) and not np.isnan(ema26[i]):
            macd_line[i] = ema12[i] - ema26[i]
    
    # Signal line (9-period EMA of MACD)
    signal_line = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 35:  # Need 26+9 for signal
        # First 9 values of MACD line for initial EMA
        macd_valid = macd_line[26:]  # Start from where MACD is valid
        if len(macd_valid) >= 9:
            signal_line[34] = np.mean(macd_valid[0:9])  # Index 34 = 26+8
            for i in range(35, len(close_1d)):
                idx = i - 26  # MACD index
                signal_line[i] = (macd_line[i] * 2 / 10) + (signal_line[i-1] * 8 / 10)
    
    # MACD histogram
    macd_hist = np.full(len(close_1d), np.nan)
    for i in range(len(close_1d)):
        if not np.isnan(macd_line[i]) and not np.isnan(signal_line[i]):
            macd_hist[i] = macd_line[i] - signal_line[i]
    
    macd_hist_aligned = align_htf_to_ltf(prices, df_1d, macd_hist)
    
    # 1-day RSI(14) for mean-reversion
    if len(close_1d) >= 14:
        delta = np.diff(close_1d)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(len(close_1d), np.nan)
        avg_loss = np.full(len(close_1d), np.nan)
        
        avg_gain[13] = np.mean(gain[0:14])
        avg_loss[13] = np.mean(loss[0:14])
        
        for i in range(14, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
        
        rs = np.full(len(close_1d), np.nan)
        rsi = np.full(len(close_1d), np.nan)
        
        for i in range(13, len(close_1d)):
            if avg_loss[i] != 0:
                rs[i] = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - (100 / (1 + rs[i]))
            else:
                rsi[i] = 100 if avg_gain[i] > 0 else 0
    else:
        rsi = np.full(len(close_1d), np.nan)
    
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # 1-week volume average for confirmation
    df_1w = get_htf_data(prices, '1w')
    vol_1w = df_1w['volume'].values
    vol_ma_1w = np.full(len(vol_1w), np.nan)
    
    for i in range(4, len(vol_1w)):  # 5-period average
        vol_ma_1w[i] = np.mean(vol_1w[i-4:i+1])
    
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(35, 13, 4)  # MACD needs 35, RSI needs 13, volume needs 4
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(macd_hist_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.3x weekly average
        volume_filter = volume[i] > vol_ma_aligned[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: RSI overbought or MACD histogram negative or stoploss
            if (rsi_aligned[i] > 70 or 
                macd_hist_aligned[i] < 0 or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: RSI oversold or MACD histogram positive or stoploss
            if (rsi_aligned[i] < 30 or 
                macd_hist_aligned[i] > 0 or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if volume_filter:
                # Long: RSI oversold and MACD histogram turning positive
                if (rsi_aligned[i] < 30 and 
                    macd_hist_aligned[i] > macd_hist_aligned[i-1]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: RSI overbought and MACD histogram turning negative
                elif (rsi_aligned[i] > 70 and 
                      macd_hist_aligned[i] < macd_hist_aligned[i-1]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals