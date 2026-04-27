#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA direction with RSI momentum and chop regime filter
# KAMA adapts to market noise, reducing false signals in choppy markets.
# RSI provides momentum confirmation, while chop filter avoids trend-following in ranging markets.
# Works in bull/bear by using KAMA for trend direction and RSI for entry timing.
# Target: 30-100 total trades over 4 years (~7-25/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for KAMA and chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average ) on 1d
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=1)
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smoothing constants
    sc = (er * (2/(2+2) - 2/(30+2)) + 2/(30+2)) ** 2
    # KAMA calculation
    kama_1d = np.full_like(close_1d, np.nan)
    kama_1d[9] = close_1d[9]  # seed
    for i in range(10, len(close_1d)):
        kama_1d[i] = kama_1d[i-1] + sc[i] * (close_1d[i] - kama_1d[i-1])
    
    # Align KAMA to 1d timeframe (wait for 1d close)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate RSI(14) on 1d
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate Choppiness Index on 1d
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(1, len(df_1d)):
        tr = max(
            df_1d['high'].iloc[i] - df_1d['low'].iloc[i],
            abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]),
            abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1])
        )
        atr_1d[i] = tr
    # Smooth ATR
    atr_smoothed = np.full_like(atr_1d, np.nan)
    for i in range(13, len(atr_1d)):
        atr_smoothed[i] = np.mean(atr_1d[i-12:i+1])
    # Sum of true ranges over 14 periods
    tr_sum = np.full_like(atr_1d, np.nan)
    for i in range(13, len(atr_1d)):
        tr_sum[i] = np.sum(atr_1d[i-12:i+1])
    # Max and min close over 14 periods
    max_close = np.full_like(close_1d, np.nan)
    min_close = np.full_like(close_1d, np.nan)
    for i in range(13, len(close_1d)):
        max_close[i] = np.max(close_1d[i-12:i+1])
        min_close[i] = np.min(close_1d[i-12:i+1])
    # Choppiness Index
    chop_1d = np.full_like(close_1d, np.nan)
    for i in range(13, len(close_1d)):
        if tr_sum[i] > 0 and max_close[i] != min_close[i]:
            chop_1d[i] = 100 * np.log10(tr_sum[i] / (max_close[i] - min_close[i])) / np.log10(14)
        else:
            chop_1d[i] = 50.0
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume filter: volume > 1.5 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 1d data (14 bars for indicators)
    start_idx = 14
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: significant volume
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend from KAMA
        bullish_trend = price > kama_aligned[i]
        bearish_trend = price < kama_aligned[i]
        
        # Chop regime: chop > 61.8 = ranging (mean revert), chop < 38.2 = trending
        chopping = chop_aligned[i] > 61.8
        trending = chop_aligned[i] < 38.2
        
        if position == 0:
            # Long: price > KAMA, RSI > 50, in trending regime, volume confirmation
            if bullish_trend and rsi_aligned[i] > 50 and trending and vol_filter:
                signals[i] = size
                position = 1
            # Short: price < KAMA, RSI < 50, in trending regime, volume confirmation
            elif bearish_trend and rsi_aligned[i] < 50 and trending and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price < KAMA or RSI < 40 or chop > 61.8 (ranging)
            if price <= kama_aligned[i] or rsi_aligned[i] < 40 or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price > KAMA or RSI > 60 or chop > 61.8 (ranging)
            if price >= kama_aligned[i] or rsi_aligned[i] > 60 or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_RSI_Chop_Regime"
timeframe = "1d"
leverage = 1.0