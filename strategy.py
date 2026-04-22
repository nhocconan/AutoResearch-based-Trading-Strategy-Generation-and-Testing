#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA direction + 1d RSI with volume spike and chop filter
# Long when KAMA rising, RSI < 30 (oversold), volume > 1.5x 20-period avg, chop > 61.8 (range)
# Short when KAMA falling, RSI > 70 (overbought), volume > 1.5x 20-period avg, chop > 61.8
# Exit when RSI crosses 50 or KAMA direction changes
# Designed for low trade frequency (~15-30/year) with mean reversion in ranging markets.
# Works in both bull/bear by using RSI extremes and chop regime filter.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 14-period RSI on 1d close
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.values
    
    # Align 1d RSI to 4h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate KAMA on 4h close
    close = prices['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close, k=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.zeros_like(change)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Initialize KAMA
    kama = np.zeros_like(close)
    kama[9] = close[9]  # Start at index 9 for 10-period lookback
    for i in range(10, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate chopiness index (14-period) on 4h
    high = prices['high'].values
    low = prices['low'].values
    atr = np.zeros(n)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr[1:] = np.trunc(np.mean(tr) * 100) / 100  # Simplified ATR
    # True range for chop calculation
    tr_sum = np.nansum(np.lib.stride_tricks.sliding_window_view(tr, 14), axis=1)
    max_high = np.max(np.lib.stride_tricks.sliding_window_view(high, 14), axis=1)
    min_low = np.min(np.lib.stride_tricks.sliding_window_view(low, 14), axis=1)
    chop = np.zeros(n)
    denom = max_high - min_low
    mask = denom != 0
    chop[13:] = 100 * np.log10(tr_sum[mask] / denom[mask]) / np.log10(14)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(kama[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        rsi_val = rsi_1d_aligned[i]
        kama_val = kama[i]
        chop_val = chop[i]
        kama_prev = kama[i-1] if i > 0 else kama_val
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        # Chop filter: chop > 61.8 indicates ranging market
        chop_filter = chop_val > 61.8
        # KAMA direction: rising if current > previous
        kama_rising = kama_val > kama_prev
        kama_falling = kama_val < kama_prev
        
        if position == 0:
            # Long conditions: KAMA rising, RSI oversold, volume spike, chop filter
            if kama_rising and rsi_val < 30 and vol_spike and chop_filter:
                signals[i] = 0.25
                position = 1
            # Short conditions: KAMA falling, RSI overbought, volume spike, chop filter
            elif kama_falling and rsi_val > 70 and vol_spike and chop_filter:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: RSI crosses 50 or KAMA direction changes
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when RSI crosses above 50 or KAMA turns down
                if rsi_val >= 50 or not kama_rising:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when RSI crosses below 50 or KAMA turns up
                if rsi_val <= 50 or not kama_falling:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_KAMA_RSI_Volume_Chop"
timeframe = "4h"
leverage = 1.0