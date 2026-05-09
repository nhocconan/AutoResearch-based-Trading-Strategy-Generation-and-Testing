#!/usr/bin/env python3
# Hypothesis: 4h KAMA direction + RSI(14) + Choppiness filter for trend-following in trending markets and mean-reversion in ranging markets
# Uses 1d timeframe for Choppiness Index regime filter (chop > 61.8 = range, chop < 38.2 = trend)
# Long when: KAMA rising, RSI > 50, chop < 38.2 (trending) OR RSI < 30 (oversold in range)
# Short when: KAMA falling, RSI < 50, chop < 38.2 (trending) OR RSI > 70 (overbought in range)
# Exit when: KAMA direction reverses OR RSI crosses 50 in trending mode OR RSI exits extreme in ranging mode
# Position size: 0.25 to balance risk and return. Target: 30-60 trades/year.

name = "4h_KAMA_RSI_Chop_Regime"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average) parameters
    er_len = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_len))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Handle first er_len values
    er = np.full_like(change, np.nan, dtype=np.float64)
    er[er_len-1:] = change[er_len-1:] / volatility[er_len-1:]
    # Fill beginning with 0
    er = np.concatenate([np.full(er_len-1, 0.0), er])
    
    # Smoothing Constant (SC)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.full_like(close, np.nan, dtype=np.float64)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    kama_prev = np.roll(kama, 1)
    kama_prev[0] = kama[0]
    kama_rising = kama > kama_prev
    kama_falling = kama < kama_prev
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(gain, np.nan, dtype=np.float64)
    avg_loss = np.full_like(loss, np.nan, dtype=np.float64)
    
    # First average
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    # Wilder smoothing
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, 50.0), rsi])  # First 14 values as 50
    
    # Get 1d data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range (TR) for Choppiness
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([np.array([high_1d[0] - low_1d[0]]), tr])  # First TR
    
    # Sum of TR over 14 periods
    tr_sum = np.convolve(tr, np.ones(14), mode='valid')
    tr_sum = np.concatenate([np.full(13, np.nan), tr_sum])
    
    # Max(high) - Min(low) over 14 periods
    max_high = np.zeros_like(high_1d)
    min_low = np.zeros_like(low_1d)
    for i in range(len(high_1d)):
        start_idx = max(0, i - 13)
        end_idx = i + 1
        max_high[i] = np.max(high_1d[start_idx:end_idx])
        min_low[i] = np.min(low_1d[start_idx:end_idx])
    
    range_14 = max_high - min_low
    
    # Choppiness Index (CHOP)
    chop = np.full_like(range_14, np.nan, dtype=np.float64)
    valid_idx = ~np.isnan(tr_sum) & (range_14 != 0)
    chop[valid_idx] = 100 * np.log10(tr_sum[valid_idx] / range_14[valid_idx]) / np.log10(14)
    
    # Align 1d indicators to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    kama_rising_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kama_rising.astype(float))
    kama_falling_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kama_falling.astype(float))
    rsi_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(kama_rising_aligned[i]) or
            np.isnan(kama_falling_aligned[i]) or np.isnan(rsi_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop_aligned[i]
        kama_rise = kama_rising_aligned[i]
        kama_fall = kama_falling_aligned[i]
        rsi_val = rsi_aligned[i]
        
        if position == 0:
            # Enter long: trending up OR oversold in range
            if ((chop_val < 38.2 and kama_rise and rsi_val > 50) or  # Trending up
                (chop_val >= 38.2 and rsi_val < 30)):  # Oversold in range
                signals[i] = 0.25
                position = 1
            # Enter short: trending down OR overbought in range
            elif ((chop_val < 38.2 and kama_fall and rsi_val < 50) or  # Trending down
                  (chop_val >= 38.2 and rsi_val > 70)):  # Overbought in range
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend reversal OR RSI crosses 50 in trend OR exits oversold in range
            if ((chop_val < 38.2 and not kama_rise) or  # Trend down
                (chop_val < 38.2 and rsi_val < 50) or  # RSI crosses below 50 in trend
                (chop_val >= 38.2 and rsi_val > 30)):  # Exits oversold in range
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend reversal OR RSI crosses 50 in trend OR exits overbought in range
            if ((chop_val < 38.2 and not kama_fall) or  # Trend up
                (chop_val < 38.2 and rsi_val > 50) or  # RSI crosses above 50 in trend
                (chop_val >= 38.2 and rsi_val < 70)):  # Exits overbought in range
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals