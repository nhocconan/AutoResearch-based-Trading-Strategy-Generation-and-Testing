#!/usr/bin/env python3
"""
4h_1d_RSI_Reversal_Camarilla
Hypothesis: RSI mean reversion at extreme levels (RSI < 30 or > 70) combined with 
price rejection at Camarilla support/resistance levels and volume confirmation.
Works in both bull and bear markets by fading extremes at key levels.
Long: RSI < 30, price > S1, volume > average, price closes above open
Short: RSI > 70, price < R1, volume > average, price closes below open
Exit when RSI returns to neutral zone (40-60) or price hits opposite level.
Target: 25-40 trades/year (100-160 over 4 years) to minimize fee drag.
"""

name = "4h_1d_RSI_Reversal_Camarilla"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for RSI and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    open_4h = prices['open'].values
    volume_4h = prices['volume'].values
    
    # --- 1d RSI (14-period) ---
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    for i in range(15, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d[:14] = np.nan  # Not enough data
    
    # Align RSI to 4h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # --- 1d Camarilla Pivots (from previous day) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift to get previous day's data for today's pivots
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels
    R1 = prev_close + ((prev_high - prev_low) * 1.0833)
    S1 = prev_close - ((prev_high - prev_low) * 1.0833)
    R3 = prev_close + ((prev_high - prev_low) * 1.2500)
    S3 = prev_close - ((prev_high - prev_low) * 1.2500)
    
    # Align pivots to 4h timeframe
    R1_4h = align_htf_to_ltf(prices, df_1d, R1)
    S1_4h = align_htf_to_ltf(prices, df_1d, S1)
    R3_4h = align_htf_to_ltf(prices, df_1d, R3)
    S3_4h = align_htf_to_ltf(prices, df_1d, S3)
    
    # --- Volume Confirmation: 4h volume > 20-period average ---
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 30  # for RSI and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(R1_4h[i]) or np.isnan(S1_4h[i]) or
            np.isnan(R3_4h[i]) or np.isnan(S3_4h[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_ok = volume_4h[i] > vol_ma_20[i]
        
        # Price action: bullish/bearish candle
        bullish_candle = close_4h[i] > open_4h[i]
        bearish_candle = close_4h[i] < open_4h[i]
        
        if position == 0:
            # Long setup: RSI oversold, price at support, bullish candle, volume
            if (rsi_1d_aligned[i] < 30 and 
                close_4h[i] > S1_4h[i] and 
                bullish_candle and 
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Short setup: RSI overbought, price at resistance, bearish candle, volume
            elif (rsi_1d_aligned[i] > 70 and 
                  close_4h[i] < R1_4h[i] and 
                  bearish_candle and 
                  vol_ok):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: RSI returns to neutral or price hits resistance
                if (rsi_1d_aligned[i] >= 40 and rsi_1d_aligned[i] <= 60) or close_4h[i] >= R1_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: RSI returns to neutral or price hits support
                if (rsi_1d_aligned[i] >= 40 and rsi_1d_aligned[i] <= 60) or close_4h[i] <= S1_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals