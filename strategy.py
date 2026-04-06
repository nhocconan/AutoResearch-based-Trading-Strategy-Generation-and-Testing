#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversal with 1d EMA(50) filter and volume confirmation
# Long when price closes below S3 and reverses above S2 with volume > 1.5x average and price > 1d EMA(50)
# Short when price closes above R3 and reverses below R2 with volume > 1.5x average and price < 1d EMA(50)
# Exit when price reaches opposite Camarilla level (S4 for long, R4 for short) or stops at 2*ATR
# Position size: 0.25
# Uses Camarilla levels from daily pivot for mean reversion in ranging markets
# Works in both bull/bear by filtering with 1d EMA(50) trend

name = "6h_camarilla_1d_ema_vol_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    # Using close of previous day (shifted by 1 to avoid look-ahead)
    if len(close_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC
    prev_close = close_1d[:-1]  # t-1
    prev_high = high_1d[:-1]    # t-1
    prev_low = low_1d[:-1]      # t-1
    
    # Calculate pivot and ranges
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels
    s1 = close_1d[:-1] - (range_val * 1.0833)
    s2 = close_1d[:-1] - (range_val * 1.1666)
    s3 = close_1d[:-1] - (range_val * 1.2500)
    s4 = close_1d[:-1] - (range_val * 1.5000)
    
    r1 = close_1d[:-1] + (range_val * 1.0833)
    r2 = close_1d[:-1] + (range_val * 1.1666)
    r3 = close_1d[:-1] + (range_val * 1.2500)
    r4 = close_1d[:-1] + (range_val * 1.5000)
    
    # Align Camarilla levels to 6h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    
    # 1d EMA(50) for trend filter
    if len(close_1d) < 50:
        return np.zeros(n)
    
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation - use 1d average volume
    volume_1d = df_1d['volume'].values
    if len(volume_1d) < 20:
        return np.zeros(n)
    
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(s2_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s4_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_ma_1d_aligned[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price reaches S4 (target) or breaks below S3 (stop)
            elif close[i] <= s4_aligned[i] or close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price reaches R4 (target) or breaks above R3 (stop)
            elif close[i] >= r4_aligned[i] or close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for reversal entries with volume confirmation
            # Long: price was below S3 (oversold) and now reverses above S2
            # Only in uptrend (price > EMA) or ranging market
            if (close[i-1] <= s3_aligned[i] and  # Was at or below S3
                close[i] > s2_aligned[i] and      # Now above S2
                volume[i] > 1.5 * volume_ma_1d_aligned[i] and  # Volume confirmation
                (close[i] > ema_1d_aligned[i] or abs(close[i] - ema_1d_aligned[i]) < 0.5 * atr[i])):  # Not strongly bearish
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price was above R3 (overbought) and now reverses below R2
            # Only in downtrend (price < EMA) or ranging market
            elif (close[i-1] >= r3_aligned[i] and   # Was at or above R3
                  close[i] < r2_aligned[i] and      # Now below R2
                  volume[i] > 1.5 * volume_ma_1d_aligned[i] and  # Volume confirmation
                  (close[i] < ema_1d_aligned[i] or abs(close[i] - ema_1d_aligned[i]) < 0.5 * atr[i])):  # Not strongly bullish
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals